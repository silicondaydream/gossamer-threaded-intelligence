"""
The ``CoordinationPrimitive`` interface: the swappable coordination-algorithm
abstraction the DCC trilogy compares head-to-head (P1).

Each primitive wraps an existing coordination algorithm behind one uniform
``act(pos, vel, dt, ctx, params) -> (accel, telemetry)`` surface, so a sweep
can grid ``primitive × delay/τ × N × budget × seed`` (§2.2). Two *references*
bracket every comparison:

* :class:`NoCommReference` — agents act on own state only (comm off): the
  lower bound of what coordination buys.
* ``uses_comm`` primitives run at ``delay=0`` are the *full-comm* upper
  reference — same primitive, zero delay (a flag on the run, not a class).

This gossamer-side interface is the canonical, unit-tested definition the
papers cite. The Maneuver.Map runner does **not** route its hot path through
these wrappers — it keeps the vectorized seam in ``policies.py`` — but the
runner's registry is locked to *equivalence* with these implementations by the
test suite (as ``test_policies.py`` already does for DMB density). ``act``
returns accelerations, matching the engine's ``step(actions)`` contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.algorithms.coordination.dmb import DMBParams, dmb_step
from gossamer.tasks.base import TaskContext


def _accel_from_step(vel: np.ndarray, new_vel: np.ndarray, dt: float) -> np.ndarray:
    """Recover the acceleration a ``*_step`` applied: ``(v' - v) / dt``.

    The engine integrates ``step(actions)`` as accelerations, whereas the
    gossamer ``*_step`` functions return the post-integration velocity. This
    bridges the two so a primitive is a drop-in action source.
    """
    if dt <= 0.0:
        return np.zeros_like(vel)
    return (new_vel - vel) / dt


class CoordinationPrimitive(ABC):
    """A swappable coordination algorithm scored on the unified task substrate."""

    name: str = "base"
    uses_comm: bool = True

    def reset(self, num_agents: int, rng: np.random.Generator) -> None:
        """Initialise per-run internal state. Default: stateless (no-op)."""

    @abstractmethod
    def act(self, pos: np.ndarray, vel: np.ndarray, dt: float,
            ctx: TaskContext, params: Dict[str, Any]
            ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return ``(accel (N,3), telemetry dict)`` for the current state."""
        ...


class FlockingPrimitive(CoordinationPrimitive):
    """Classic Boids. The ``uses_comm=False`` baseline (local sensing only)."""

    name = "flocking"
    uses_comm = False

    def act(self, pos, vel, dt, ctx, params):
        p = params or {}
        new_pos, new_vel = flock_step(
            pos, vel, dt,
            alignment_weight=p.get("alignment_weight", 1.0),
            cohesion_weight=p.get("cohesion_weight", 1.0),
            separation_weight=p.get("separation_weight", 1.5),
            neighbor_radius=p.get("neighbor_radius", 10.0),
            separation_distance=p.get("separation_distance", 1.0),
            max_speed=p.get("max_speed", 5.0),
        )
        return _accel_from_step(vel, new_vel, dt), {}


class NoCommReference(CoordinationPrimitive):
    """Lower reference: no inter-agent communication at all.

    Agents merely damp toward their own current heading (coast + mild braking).
    Any coordinating primitive should beat this on every task; it is the P1
    "coordination buys nothing" floor.
    """

    name = "no_comm"
    uses_comm = False

    def act(self, pos, vel, dt, ctx, params):
        damping = (params or {}).get("damping", 0.05)
        return -damping * vel, {}


class DMBPrimitive(CoordinationPrimitive):
    """Density-Modulated Boids (§1.3 DMB), wrapped as a primitive."""

    name = "dmb"
    uses_comm = False

    def __init__(self):
        self._rng: Optional[np.random.Generator] = None

    def reset(self, num_agents, rng):
        self._rng = rng

    def act(self, pos, vel, dt, ctx, params):
        p = params or {}
        dmb_kwargs = p.get("dmb_params", {})
        dparams = DMBParams(**dmb_kwargs) if dmb_kwargs else DMBParams()
        new_pos, new_vel, info = dmb_step(
            pos, vel, dt, dparams,
            neighbor_radius=p.get("neighbor_radius", 10.0),
            separation_distance=p.get("separation_distance", 1.0),
            max_speed=p.get("max_speed", 5.0),
            max_accel=p.get("max_accel"),
            sensing_noise=p.get("sensing_noise", 0.0),
            rng=self._rng,
        )
        telemetry = {"density": info.get("density")} if "density" in info else {}
        return _accel_from_step(vel, new_vel, dt), telemetry


class GossipConsensusPrimitive(CoordinationPrimitive):
    """Decentralized average-consensus over the local (radius) graph.

    One synchronous consensus step per ``act`` mixes each agent's *estimate*
    (its position) with its radius-neighbours' estimates, then steers toward the
    mixed estimate. This is the ``uses_comm=True`` middle ground between local
    flocking and CRDT-intent, and the natural partner of
    :class:`~gossamer.tasks.tasks.ConsensusTask`.

    Kept O(N·k) via a spatial grid (no dense adjacency) so it scales with the
    rest of the seam; it is the algebraic sibling of
    :func:`gossamer.algorithms.coordination.consensus.average_consensus`.
    """

    name = "gossip"
    uses_comm = True

    def act(self, pos, vel, dt, ctx, params):
        from gossamer.utils.spatial import build_grid, neighbors_within

        p = params or {}
        radius = float(p.get("neighbor_radius", 10.0))
        gain = float(p.get("consensus_gain", 1.0))
        max_speed = float(p.get("max_speed", 5.0))

        n = pos.shape[0]
        if n == 0:
            return np.zeros_like(pos), {}
        grid, cell_idx = build_grid(pos, radius)
        target = pos.copy()
        for i in range(n):
            cand = neighbors_within(pos, cell_idx, grid, radius, i)
            if not cand:
                continue
            cand = np.asarray(cand, dtype=int)
            # Mix own estimate with the neighbourhood mean (one gossip round).
            nbr_mean = pos[cand].mean(axis=0)
            target[i] = 0.5 * pos[i] + 0.5 * nbr_mean
        desired_vel = gain * (target - pos)
        speed = np.linalg.norm(desired_vel, axis=1, keepdims=True)
        clip = np.minimum(1.0, max_speed / np.maximum(speed, 1e-9))
        desired_vel = desired_vel * clip
        accel = _accel_from_step(vel, desired_vel, dt)
        return accel, {}


# Registry mirroring ``benchmarks.scenarios.ALL_SCENARIOS`` and
# ``tasks.tasks.ALL_TASKS``. CRDT-intent and HMA-market are exercised through
# the runner's stateful seam (they need per-run CRDT / auction state that the
# runner already carries); they are intentionally not in this stateless-friendly
# registry to avoid two divergent copies of their orchestration.
PRIMITIVES: Dict[str, Callable[[], CoordinationPrimitive]] = {
    "flocking": FlockingPrimitive,
    "no_comm": NoCommReference,
    "dmb": DMBPrimitive,
    "gossip": GossipConsensusPrimitive,
}


__all__ = [
    "CoordinationPrimitive",
    "DMBPrimitive",
    "FlockingPrimitive",
    "GossipConsensusPrimitive",
    "NoCommReference",
    "PRIMITIVES",
]
