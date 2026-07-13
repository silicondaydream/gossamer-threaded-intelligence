"""Classical (non-learned) comparator policies — the DMB Table-1 references.

These are the structure-free and myopic baselines the density-modulated policy is
measured against: a Lévy-flight search that never communicates, and a greedy
nearest-unvisited-cell coverage heuristic that has no stigmergy. They are the
denominators of the DMB claims, so they are algorithms in exactly the sense the
architecture means it, and they belong here rather than in the runner.

They lived in `maneuver-map/backend/app/policies.py` as `_act_levy` / `_act_greedy`
— real algorithm code in the orchestration layer, which is the defect class the
Gossamer extraction was supposed to end and which quietly regrew. Moved here
BYTE-FOR-BYTE: same RNG draw order, same reduction order, same window scan. The
24-cell coordination fingerprint hashes both primitives, so the move is verifiable
rather than merely plausible — and it must stay a move. Reordering the arithmetic
of a migration is how the HMA headline shifted (a scan over distances became a
reduction over squared distances; the fp ties flipped; a published number changed).

The learned comparators (MAPPO relay selection, learned Boids weights) live in
`gossamer.learning.baselines` — separate because they carry a torch dependency
these deliberately do not.

Each policy carries its own small state object rather than a bag of keys in a
dict: the state IS part of the algorithm (a Lévy walk without its reorientation
countdown is not a Lévy walk), so it travels with it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "LevyState",
    "levy_flight_accel",
    "CoverageGrid",
    "greedy_coverage_accel",
]


# ---------------------------------------------------------------------------
# Lévy-flight search (structure-free, non-communicating)
# ---------------------------------------------------------------------------
@dataclass
class LevyState:
    """The RNG stream and per-agent reorientation countdown of a Lévy walk.

    Seeded, and the seed is the whole reproducibility story for this primitive:
    the walk is pure noise, so an unseeded stream would make it unreplayable and
    every number it produces unfalsifiable.
    """
    rng: np.random.Generator
    countdown: np.ndarray  # (N,) steps remaining until each agent reorients

    @classmethod
    def create(cls, n: int, seed: int = 12345) -> "LevyState":
        return cls(rng=np.random.default_rng(int(seed)), countdown=np.zeros(n))


def levy_flight_accel(
    vel: np.ndarray,
    dt: float,
    state: LevyState,
    *,
    max_speed: float = 5.0,
    mu: float = 2.0,
    mean_reorient_steps: float = 8.0,
) -> np.ndarray:
    """Lévy-flight biological-search baseline (DMB Table-1 comparator).

    A *non-communicating* search policy: each agent persists on its heading and,
    at exponentially-spaced reorientation events, redraws an isotropic heading and
    a heavy-tailed (Lévy, exponent ``mu``) step speed. No neighbour interaction —
    this is the "structure-free search" reference DMB is scored against on ψ /
    collisions / coverage. Agents between reorientations return zero accel, so the
    engine integrates a straight flight.

    ``mu ≈ 2`` is the optimal-search exponent (Viswanathan et al.); it is a
    parameter rather than a constant because the claim "DMB beats optimal Lévy
    search" is only worth making if the baseline was actually given its optimum.

    Mutates ``state`` (draws from the RNG, decrements the countdown).
    """
    n = vel.shape[0]
    if n == 0:
        return np.zeros_like(vel)
    rng, cd = state.rng, state.countdown
    due = cd <= 0.0
    tgt = vel.copy()
    if np.any(due):
        k = int(due.sum())
        dirs = rng.normal(size=(k, 3))
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)
        # Heavy-tailed (Pareto/Lévy) step lengths, normalised into [0, max_speed].
        u = rng.random(k)
        step = (1.0 - u) ** (-1.0 / max(1e-3, mu - 1.0))
        smax = float(step.max())
        step = step / smax * max_speed if smax > 0 else step
        tgt[due] = dirs * step[:, None]
        cd[due] = rng.exponential(mean_reorient_steps, size=k)
    cd -= 1.0
    return (tgt - vel) / max(dt, 1e-9)


# ---------------------------------------------------------------------------
# Greedy nearest-unvisited coverage (myopic, no stigmergy)
# ---------------------------------------------------------------------------
@dataclass
class CoverageGrid:
    """A dense boolean visitation grid — greedy coverage's entire memory.

    Deliberately NOT a pheromone field: there is no decay and no deposit mass,
    which is the point. Greedy is what stigmergy is worth *over*, so giving it any
    of TF-ACO's machinery would flatter it and blunt the comparison.
    """
    visited: np.ndarray  # (grid_h, grid_w) bool
    bound: float

    @classmethod
    def create(cls, resolution: int = 64, bound: float = 100.0) -> "CoverageGrid":
        r = int(resolution)
        return cls(visited=np.zeros((r, r), dtype=bool), bound=float(bound))

    def coverage(self) -> float:
        if self.visited.size == 0:
            return 0.0
        return float(np.count_nonzero(self.visited) / self.visited.size)


def greedy_coverage_accel(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    grid: CoverageGrid,
    *,
    max_speed: float = 5.0,
    max_accel: float = 1.0,
    window: int = 3,
) -> np.ndarray:
    """Greedy nearest-unvisited-cell baseline (DMB Table-1 coverage comparator).

    Each agent steers toward the nearest *unvisited* coverage cell in a local
    window — the myopic "go to the closest unserved task" heuristic that TF-ACO's
    stigmergy is compared against. Same dense-grid bookkeeping as
    :func:`tfaco_accel`, minus the pheromone field: pure distance-greedy, no
    revisit decay, no message passing. Where the local window is fully visited the
    agent coasts, which is how it escapes a cluster it has already covered.

    Mutates ``grid`` in place. Returns the acceleration (N, 3).
    """
    n = pos.shape[0]
    if n == 0:
        return np.zeros_like(vel)
    b = float(grid.bound)
    vis = grid.visited
    gh, gw = vis.shape
    cx = np.clip(((pos[:, 0] + b) / (2 * b) * gw).astype(np.intp), 0, gw - 1)
    cy = np.clip(((pos[:, 1] + b) / (2 * b) * gh).astype(np.intp), 0, gh - 1)
    vis[cy, cx] = True
    w = int(window)
    best_d = np.full(n, np.inf)
    best_jx = cx.copy()
    best_jy = cy.copy()
    for dy in range(-w, w + 1):
        for dx in range(-w, w + 1):
            jy = np.clip(cy + dy, 0, gh - 1)
            jx = np.clip(cx + dx, 0, gw - 1)
            dist = np.hypot(jx - cx, jy - cy)
            score = np.where(~vis[jy, jx], dist, np.inf)  # only unvisited count
            better = score < best_d
            best_d = np.where(better, score, best_d)
            best_jx = np.where(better, jx, best_jx)
            best_jy = np.where(better, jy, best_jy)
    tx = (best_jx + 0.5) / gw * (2 * b) - b
    ty = (best_jy + 0.5) / gh * (2 * b) - b
    dirx = tx - pos[:, 0]
    diry = ty - pos[:, 1]
    norm = np.sqrt(dirx * dirx + diry * diry) + 1e-9
    new_vel = vel.copy()
    new_vel[:, 0] += (dirx / norm) * max_accel * dt
    new_vel[:, 1] += (diry / norm) * max_accel * dt
    # Saturated window (nothing unvisited nearby): coast to escape the cluster.
    saturated = ~np.isfinite(best_d)
    if np.any(saturated):
        new_vel[saturated] = vel[saturated]
    spd = np.linalg.norm(new_vel, axis=1)
    fast = spd > max_speed
    if np.any(fast):
        new_vel[fast] = new_vel[fast] / spd[fast, None] * max_speed
    return (new_vel - vel) / max(dt, 1e-9)
