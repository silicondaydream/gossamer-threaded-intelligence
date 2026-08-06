"""
Standard baselines for the benchmark suite.

Every scenario gets evaluated against the same set, so claims of
"algorithm X beats classical" can be audited. Each baseline is a
callable returning ``(num_agents, 3)`` accelerations given the current
state. Stateful baselines return a closure over their internal state;
stateless ones are plain functions.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from gossamer.algorithms.coordination.flocking import flock_step


Baseline = Callable[[np.ndarray, np.ndarray, np.random.Generator], np.ndarray]


def random_baseline(scale: float = 1.0) -> Baseline:
    """Uniform random accelerations — the "chance" lower bound."""
    def _f(pos, vel, rng):
        return rng.uniform(-scale, scale, size=pos.shape)
    return _f


def do_nothing_baseline() -> Baseline:
    def _f(pos, vel, rng):
        return np.zeros_like(pos)
    return _f


def greedy_rendezvous() -> Baseline:
    """Go-to-centroid greedy — optimal for the rendezvous scenario."""
    def _f(pos, vel, rng):
        centroid = pos.mean(axis=0, keepdims=True)
        direction = centroid - pos
        norm = np.linalg.norm(direction, axis=1, keepdims=True) + 1e-9
        return direction / norm
    return _f


def greedy_disperse(neighbor_radius: float = 10.0) -> Baseline:
    """Push away from nearest neighbor — greedy disperse solution."""
    def _f(pos, vel, rng):
        n = pos.shape[0]
        if n < 2:
            return np.zeros_like(pos)
        diff = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        nearest = np.argmin(d, axis=1)
        away = pos - pos[nearest]
        norm = np.linalg.norm(away, axis=1, keepdims=True) + 1e-9
        return away / norm
    return _f


def gossamer_flocking(
    alignment: float = 1.0,
    cohesion: float = 1.0,
    separation: float = 1.5,
    neighbor_radius: float = 10.0,
    separation_distance: float = 1.0,
    max_speed: float = 5.0,
    dt: float = 0.1,
) -> Baseline:
    """Classical Boids as a per-step acceleration baseline."""
    def _f(pos, vel, rng):
        new_pos, new_vel = flock_step(
            pos, vel, dt,
            alignment_weight=alignment,
            cohesion_weight=cohesion,
            separation_weight=separation,
            neighbor_radius=neighbor_radius,
            separation_distance=separation_distance,
            max_speed=max_speed,
            use_spatial=True,
        )
        return (new_vel - vel) / max(dt, 1e-9)
    return _f


def vicsek_alignment(speed: float = 1.0, neighbor_radius: float = 10.0,
                     dt: float = 0.1) -> Baseline:
    """Constant-speed heading alignment — the Vicsek update, as a baseline.

    Each agent steers toward the mean HEADING of its neighbours while holding
    ``|v| = speed``. That constant-speed constraint is Vicsek's defining
    property and the reason this exists separately from
    :func:`gossamer_flocking`: Boids averages velocity *vectors* (magnitude
    decays and is refilled by the cohesion term) and pulls every agent toward a
    common centre, which makes headings point inward from all directions —
    anti-polar by symmetry. A cohesive swarm and a polarised flock are
    different states, and only the second one has an order-disorder
    transition to validate a substrate against. Measured on `ReferenceEngine`:
    Boids never leaves psi ~ 0.05 at ANY alignment gain, so it cannot serve as
    the ordering instrument.

    Returns the acceleration that lands the velocity on the desired heading in
    one step, so the engine's integration re-anchors speed every step rather
    than letting it drift.

    ⚠️ Related footgun, in `flock_step` rather than here: its weights are
    dimensionless relaxation GAINS applied with no ``dt`` factor
    (``v += w*(avg - v)``), so ``alignment_weight`` 1.0 snaps exactly to the
    neighbour mean and **anything >= 2 overshoots and oscillates**. Turning
    alignment "up" to get more order gets less: psi 0.22 at gain 0.3 vs 0.008
    at gain 3.0.
    """
    from scipy.spatial import cKDTree

    def _f(pos, vel, rng):
        speeds = np.linalg.norm(vel, axis=1, keepdims=True)
        headings = vel / np.maximum(speeds, 1e-9)
        tree = cKDTree(pos)
        neighbours = tree.query_ball_point(pos, neighbor_radius)
        mean_heading = np.empty_like(headings)
        for i, idx in enumerate(neighbours):
            # `idx` always contains i itself, so an isolated agent keeps its
            # own heading rather than dividing by zero.
            mean_heading[i] = headings[idx].mean(axis=0)
        norm = np.linalg.norm(mean_heading, axis=1, keepdims=True)
        # A neighbourhood whose headings cancel exactly leaves no defined
        # direction; keep the agent's own rather than inventing one.
        desired = np.where(norm > 1e-9, mean_heading / np.maximum(norm, 1e-9),
                           headings) * speed
        return (desired - vel) / max(dt, 1e-9)
    return _f


def coverage_walker(noise_scale: float = 0.5) -> Baseline:
    """Persistent random walk — simple coverage strategy.

    Keeps per-agent heading and perturbs it lightly each step. Better
    than pure random on coverage because agents don't stop moving.

    The heading state used to be keyed on ``id(pos)``. CPython recycles the id of
    a freed object, so a later run whose position array landed at the same address
    silently inherited the PREVIOUS run's headings — a cross-run state leak that
    would show up as an unreproducible coverage number. The closure holds one
    heading array instead; a fresh baseline is built per run (``DEFAULT_BASELINES``
    maps to factories), so per-closure state is per-run state.
    """
    headings: dict = {"h": None}

    def _f(pos, vel, rng):
        h = headings["h"]
        if h is None or h.shape != pos.shape:
            h = rng.normal(size=pos.shape)
            h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        # Small heading perturbation
        h = h + rng.normal(scale=noise_scale, size=pos.shape)
        h /= np.linalg.norm(h, axis=1, keepdims=True) + 1e-9
        headings["h"] = h
        return h
    return _f


DEFAULT_BASELINES = {
    "random": lambda scenario: random_baseline(scale=1.0),
    "gossamer_flocking": lambda scenario: gossamer_flocking(),
    # The constant-speed aligner. It is the ONLY baseline that orders the
    # `vicsek_transition` row (Boids is anti-polar there — see that scenario),
    # so leaving it out would make the substrate-validation row report a flat
    # null for every baseline and look like a clean tie.
    "vicsek_alignment": lambda scenario: vicsek_alignment(),
    # Per-scenario "canonical greedy"
    "greedy": lambda scenario: {
        "dispersal": greedy_disperse(),
        "rendezvous": greedy_rendezvous(),
        "coverage": coverage_walker(),
        "leader_follower": greedy_rendezvous(),  # follow the leader via centroid proxy
        "byzantine": greedy_rendezvous(),
        "vicsek_transition": vicsek_alignment(),
    }.get(scenario.name, do_nothing_baseline()),
}


__all__ = [
    "Baseline",
    "DEFAULT_BASELINES",
    "coverage_walker",
    "do_nothing_baseline",
    "gossamer_flocking",
    "greedy_disperse",
    "greedy_rendezvous",
    "random_baseline",
    "vicsek_alignment",
]
