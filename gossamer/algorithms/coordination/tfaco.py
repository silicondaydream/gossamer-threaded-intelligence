"""
Task-Field Ant Colony Optimization (TF-ACO) for stigmergic coverage.

Agents spread out to cover a region by depositing virtual pheromone on the
cells they visit and steering toward nearby *under*-visited cells. The novelty
over a plain pheromone grid is that the field is a **CRDT**: each cell's
deposit total is a grow-only counter (:class:`gossamer.crdt.GCounter`) and the
decay clock is a last-writer-wins timestamp, so independent replicas of the map
held by different agents reconcile by :meth:`PheromoneField.merge` and converge
without locks or a leader. That makes the DMB + TF-ACO paper's "OR-Set CRDT
counters" claim concrete and ties the coverage map into the same
eventually-consistent state abstraction as ICCD intent and HMA inventory.

Effective pheromone at read time is the deposited mass decayed since the last
deposit::

    tau(cell, t) = deposits(cell) * exp(-evap_lambda * (t - last_deposit_t(cell)))

Both ``deposits`` (max-merge G-Counter) and ``last_deposit_t`` (max-merge LWW)
are join-semilattices, so the field as a whole is one too: merging is
commutative, associative, and idempotent, and replicas converge.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, Optional, Tuple

import numpy as np

from gossamer.crdt import GCounter

Cell = Tuple[int, int]


@dataclass(frozen=True)
class TFACOParams:
    grid_w: int = 64
    grid_h: int = 64
    bound: float = 500.0          # half-extent of the (square) domain in x, y
    evap_lambda: float = 0.01     # per-unit-time exponential decay rate
    deposit_rate: float = 1.0     # pheromone deposited per visit
    heuristic_weight: float = 0.5  # distance penalty in target selection
    window: int = 3               # half-width of the local target-search window


class PheromoneField:
    """A CRDT-backed sparse pheromone map over a 2-D grid.

    Internally two sparse maps keyed by cell: a :class:`GCounter` of deposits
    and a float last-deposit time. ``deposit`` mutates in place (cheap for the
    sim hot loop); ``merge`` is pure and returns a new field (for replica
    reconciliation and the convergence tests).
    """

    def __init__(self, params: TFACOParams):
        self.params = params
        self.deposits: Dict[Cell, GCounter] = {}
        self.last_t: Dict[Cell, float] = {}

    # -- geometry -------------------------------------------------------
    def cell_of(self, xy: np.ndarray) -> Cell:
        p = self.params
        b, gw, gh = p.bound, p.grid_w, p.grid_h
        cx = int(np.clip(((xy[0] + b) / (2 * b)) * gw, 0, gw - 1))
        cy = int(np.clip(((xy[1] + b) / (2 * b)) * gh, 0, gh - 1))
        return (cy, cx)

    def cell_center(self, cell: Cell) -> np.ndarray:
        p = self.params
        cy, cx = cell
        tx = (cx + 0.5) / p.grid_w * (2 * p.bound) - p.bound
        ty = (cy + 0.5) / p.grid_h * (2 * p.bound) - p.bound
        return np.array([tx, ty])

    # -- updates --------------------------------------------------------
    def deposit(self, cell: Cell, replica, t: float, amount: Optional[float] = None) -> "PheromoneField":
        amt = self.params.deposit_rate if amount is None else amount
        # GCounter is integer-valued; scale deposits to an integer quantum.
        q = max(1, int(round(amt)))
        self.deposits[cell] = self.deposits.get(cell, GCounter()).increment(replica, q)
        prev = self.last_t.get(cell, float("-inf"))
        if t > prev:
            self.last_t[cell] = float(t)
        return self

    def tau(self, cell: Cell, t: float) -> float:
        """Effective (decayed) pheromone at ``cell`` and time ``t``."""
        g = self.deposits.get(cell)
        if g is None:
            return 0.0
        age = max(0.0, t - self.last_t.get(cell, t))
        return g.value() * float(np.exp(-self.params.evap_lambda * age))

    def merge(self, other: "PheromoneField") -> "PheromoneField":
        if self.params != other.params:
            raise ValueError("cannot merge PheromoneFields with different params")
        out = PheromoneField(self.params)
        for cell in set(self.deposits) | set(other.deposits):
            a = self.deposits.get(cell)
            b = other.deposits.get(cell)
            out.deposits[cell] = a.merge(b) if (a and b) else (a or b)
        for cell in set(self.last_t) | set(other.last_t):
            out.last_t[cell] = max(self.last_t.get(cell, float("-inf")),
                                   other.last_t.get(cell, float("-inf")))
        return out

    def coverage(self) -> float:
        """Fraction of grid cells that have received at least one deposit."""
        total = self.params.grid_w * self.params.grid_h
        return len(self.deposits) / total if total else 0.0

    def dense_grid(self, t: float) -> np.ndarray:
        """Materialise the decayed field as a ``(grid_h, grid_w)`` array."""
        g = np.zeros((self.params.grid_h, self.params.grid_w), dtype=float)
        for (cy, cx) in self.deposits:
            g[cy, cx] = self.tau((cy, cx), t)
        return g


def tfaco_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    field: PheromoneField,
    t: float,
    dt: float,
    params: TFACOParams,
    *,
    max_speed: float = 5.0,
    max_accel: float = 1.0,
    replica_ids=None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """One stigmergic step: deposit, pick under-visited targets, steer, advance.

    Operates on the (x, y) ground plane; any z component of velocity is left
    unchanged so this composes with a 3-D base flocking term. Returns
    ``(new_positions, new_velocities, info)`` with per-agent pheromone reading.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    n = positions.shape[0]
    if replica_ids is None:
        replica_ids = np.arange(n)
    p = params
    per_pher = np.zeros(n, dtype=float)
    new_vel = velocities.copy()

    for a in range(n):
        cy, cx = field.cell_of(positions[a, :2])
        field.deposit((cy, cx), replica_ids[a], t)
        per_pher[a] = field.tau((cy, cx), t)

        # Search a local window for the lowest score = pheromone + dist penalty.
        best_cell, best_score = (cy, cx), float("inf")
        for jy in range(max(0, cy - p.window), min(p.grid_h, cy + p.window + 1)):
            for jx in range(max(0, cx - p.window), min(p.grid_w, cx + p.window + 1)):
                dist = float(np.hypot(jx - cx, jy - cy))
                score = field.tau((jy, jx), t) + p.heuristic_weight * dist
                if score < best_score:
                    best_score, best_cell = score, (jy, jx)

        target = field.cell_center(best_cell)
        direction = target - positions[a, :2]
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            accel_xy = (direction / norm) * max_accel
            new_vel[a, 0] += accel_xy[0] * dt
            new_vel[a, 1] += accel_xy[1] * dt

    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    over = (speed > max_speed).ravel()
    if np.any(over):
        new_vel[over] = new_vel[over] / speed[over] * max_speed

    new_pos = positions + new_vel * dt
    info = {"pheromone": per_pher, "coverage": field.coverage()}
    return new_pos, new_vel, info
