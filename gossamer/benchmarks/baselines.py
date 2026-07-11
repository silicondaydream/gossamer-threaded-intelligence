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
    # Per-scenario "canonical greedy"
    "greedy": lambda scenario: {
        "dispersal": greedy_disperse(),
        "rendezvous": greedy_rendezvous(),
        "coverage": coverage_walker(),
        "leader_follower": greedy_rendezvous(),  # follow the leader via centroid proxy
        "byzantine": greedy_rendezvous(),
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
]
