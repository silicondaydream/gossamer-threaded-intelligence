"""
Concrete DCC benchmark tasks, each with a normalized ``coordination_quality``.

* :class:`RendezvousTask` — mutually agree *where* to meet. Purely peer-derived.
* :class:`ConsensusTask` — drive the swarm's positional variance to zero.
* :class:`FormationHoldTask` — hold (and reconfigure) a geometric formation.
* :class:`CoverageHoldTask` — hold coverage of a (drifting) target cell set.
* :class:`TrackingRendezvousTask` — gather on a goal only a few agents can see,
  which jumps on the τ clock. **The task-timescale axis.**

Each ``coordination_quality`` returns ``Q ∈ [0, 1]`` (1 = perfect). Length scales
are normalized by ``bound`` (captured at ``init_goal`` time) so quality is
comparable across domain sizes.

**Which tasks does ``tau_sec`` actually control?** Only those that (a) move their
goal in ``perturb`` *and* (b) let that goal reach the dynamics or the metric.

===================  ===========  ==================  ==================
task                 goal moves   goal enters Q/accel  τ controls Q?
===================  ===========  ==================  ==================
rendezvous           no           no                   **no**
consensus            yes          no                   **no**
formation_hold       yes          Q only               weakly
coverage_hold        yes          yes                  yes (Q ≈ 0.01)
tracking_rendezvous  yes          yes                  **yes**
===================  ===========  ==================  ==================

``rendezvous`` and ``consensus`` are τ-inert *by design*: their Q measures swarm
compactness / variance reduction, which is precisely what makes coordination
necessary and what delay degrades. This is the right design for the phase-diagram
paper — but it means ``tau_sec`` is not the timescale in "the ratio of delay to
the coordination timescale". That is the swarm's **intrinsic** convergence time,
set by the primitive's gain / neighbour radius / speed. Sweeping ``tau_sec``
against those two tasks returns a null by construction. Use
:class:`TrackingRendezvousTask` for an *imposed* task timescale.
"""
from __future__ import annotations

import numpy as np

from gossamer.tasks.base import (
    CoordinationTask,
    GoalState,
    TaskContext,
    _clip01,
    _reconfig_due,
)


# --------------------------------------------------------------------------- #
# Rendezvous
# --------------------------------------------------------------------------- #
class RendezvousTask(CoordinationTask):
    """Mutual gather: agents must agree *where* to meet, using only (delayed)
    peer positions — no externally given point, so coordination is *necessary*
    (an isolated agent cannot solve it). This is the peer-derived-target design
    that yields the clean delay collapse; contrast the individually-solvable
    home-to-known-point task, where naive coordination merely becomes a liability
    under delay.

    ``Q = exp(-mean_dist_to_centroid / L0)`` — swarm compactness, 1 when fully
    gathered. ``L0`` is a fixed fraction of the initial spread so Q is
    domain-size invariant. The primitive's cohesion (on delayed peers) drives the
    gathering; large delay makes it overshoot/oscillate and Q collapses.
    """

    name = "rendezvous"

    def __init__(self, length_scale_frac: float = 0.25):
        self.length_scale_frac = length_scale_frac

    def init_goal(self, rng, num_agents, bound):
        return GoalState(extra={"bound": float(bound),
                                "L0": float(bound) * self.length_scale_frac})

    def perturb(self, goal, ctx, rng):
        # Static target set (the meeting point is emergent); the τ axis for the
        # mutual-gather task is its own convergence timescale.
        return goal

    def coordination_quality(self, pos, vel, goal):
        if pos.shape[0] == 0:
            return 0.0
        centroid = pos.mean(axis=0, keepdims=True)
        mean_dist = float(np.linalg.norm(pos - centroid, axis=1).mean())
        L0 = goal.extra["L0"] or 1.0
        return _clip01(float(np.exp(-mean_dist / L0)))


# --------------------------------------------------------------------------- #
# Formation hold
# --------------------------------------------------------------------------- #
class FormationHoldTask(CoordinationTask):
    """Hold a per-agent offset template about a centre; reconfigure on τ.

    ``Q = 1 - clip(mean_offset_error / L0, 0, 1)`` where ``offset_error`` is the
    distance between each agent's position (relative to the swarm centroid) and
    its assigned template offset. ``perturb`` rotates the template about the
    z-axis by a τ-paced increment — the "reconfiguration rate".
    """

    name = "formation_hold"

    def __init__(self, length_scale_frac: float = 0.25, rotate_rad: float = np.pi / 6):
        self.length_scale_frac = length_scale_frac
        self.rotate_rad = rotate_rad

    def _lattice(self, num_agents, bound):
        # Deterministic ring/lattice offsets on a circle of radius ~bound*0.2.
        r = bound * 0.2
        ang = np.linspace(0.0, 2 * np.pi, num_agents, endpoint=False)
        offs = np.zeros((num_agents, 3))
        offs[:, 0] = r * np.cos(ang)
        offs[:, 1] = r * np.sin(ang)
        return offs

    def init_goal(self, rng, num_agents, bound):
        offsets = self._lattice(num_agents, bound)
        return GoalState(offsets=offsets, phase=0.0,
                         extra={"bound": float(bound),
                                "L0": float(bound) * self.length_scale_frac})

    def perturb(self, goal, ctx, rng):
        if not _reconfig_due(ctx):
            return goal
        theta = self.rotate_rad
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        new_offsets = goal.offsets @ rot.T
        return GoalState(offsets=new_offsets, phase=goal.phase + theta, extra=goal.extra)

    def coordination_quality(self, pos, vel, goal):
        if pos.shape[0] == 0:
            return 0.0
        centroid = pos.mean(axis=0, keepdims=True)
        rel = pos - centroid
        err = float(np.linalg.norm(rel - goal.offsets, axis=1).mean())
        L0 = goal.extra["L0"] or 1.0
        return _clip01(1.0 - err / L0)


# --------------------------------------------------------------------------- #
# Coverage hold
# --------------------------------------------------------------------------- #
class CoverageHoldTask(CoordinationTask):
    """Hold occupancy of a target set of grid cells; the set drifts on τ.

    ``Q = |occupied ∩ target| / |target|`` — the fraction of the target cell
    set currently occupied by at least one agent. Unlike the benchmark
    ``CoverageScenario`` (cumulative *visited*), this is an *instantaneous hold*
    metric, so degradation under delay shows up immediately.

    Cell indexing mirrors ``CoverageScenario`` / the runner's coverage grid:
    ``(iy, ix)`` on an XY grid of width ``2*bound/resolution``.
    """

    name = "coverage_hold"

    def __init__(self, grid_resolution_frac: float = 0.1, target_frac: float = 0.25):
        self.grid_resolution_frac = grid_resolution_frac
        self.target_frac = target_frac

    def _cell_width(self, bound):
        res = max(1e-9, bound * self.grid_resolution_frac)
        return max(1, int(2 * bound / res))

    def _all_cells(self, w):
        return [(iy, ix) for iy in range(w) for ix in range(w)]

    def init_goal(self, rng, num_agents, bound):
        w = self._cell_width(bound)
        all_cells = self._all_cells(w)
        k = max(1, int(len(all_cells) * self.target_frac))
        chosen = rng.choice(len(all_cells), size=k, replace=False)
        target = frozenset(all_cells[i] for i in chosen)
        return GoalState(target_cells=target,
                         extra={"bound": float(bound), "w": w})

    def perturb(self, goal, ctx, rng):
        if not _reconfig_due(ctx):
            return goal
        w = goal.extra["w"]
        all_cells = self._all_cells(w)
        k = len(goal.target_cells)
        chosen = rng.choice(len(all_cells), size=k, replace=False)
        target = frozenset(all_cells[i] for i in chosen)
        return GoalState(target_cells=target, extra=goal.extra)

    def _occupied(self, pos, bound, w):
        if pos.shape[0] == 0:
            return set()
        cx = np.clip(((pos[:, 0] + bound) / (2 * bound)) * w, 0, w - 1).astype(int)
        cy = np.clip(((pos[:, 1] + bound) / (2 * bound)) * w, 0, w - 1).astype(int)
        return set(zip(cy.tolist(), cx.tolist()))

    def coordination_quality(self, pos, vel, goal):
        target = goal.target_cells
        if not target:
            return 0.0
        occ = self._occupied(pos, goal.extra["bound"], goal.extra["w"])
        return _clip01(len(occ & target) / len(target))

    def goal_accel(self, pos, vel, goal, max_accel):
        # Pull agents toward the centre of the target region; the primitive's
        # separation (on delayed peers) is what spreads them to fill the cells,
        # so delay causes overlap/gaps and coverage drops.
        b, w = goal.extra["bound"], goal.extra["w"]
        cells = np.array(sorted(goal.target_cells), dtype=float) if goal.target_cells else None
        if cells is None or cells.shape[0] == 0:
            return np.zeros_like(pos)
        # Cell (iy, ix) → world centre on the XY plane.
        cx = (cells[:, 1] + 0.5) / w * (2 * b) - b
        cy = (cells[:, 0] + 0.5) / w * (2 * b) - b
        centre = np.array([cx.mean(), cy.mean(), 0.0])
        d = centre[None, :] - pos
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return max_accel * d / np.maximum(n, 1e-9)


# --------------------------------------------------------------------------- #
# Consensus
# --------------------------------------------------------------------------- #
class ConsensusTask(CoordinationTask):
    """Agree on a shared scalar/vector value; the target drifts on τ.

    The per-agent *estimate* is taken to be the agent's position (the primitive
    is expected to drive positions toward agreement — this is the physical
    substrate for the abstract consensus value). ``Q = 1 - clip(var / var0)``
    where ``var`` is the mean coordinate variance of the estimates and ``var0``
    is a fixed reference (set from the initial spread), so Q rises to 1 as the
    swarm converges. ``perturb`` jumps the (informational) target on the τ
    clock; the target biases nothing physical directly but is carried so P3's
    predictor has a moving quantity to anticipate.
    """

    name = "consensus"

    def __init__(self, ref_frac: float = 0.25):
        self.ref_frac = ref_frac

    def init_goal(self, rng, num_agents, bound):
        value = rng.uniform(-bound * 0.5, bound * 0.5, size=3)
        var0 = (bound * self.ref_frac) ** 2
        return GoalState(value=value, extra={"bound": float(bound), "var0": float(var0)})

    def perturb(self, goal, ctx, rng):
        if not _reconfig_due(ctx):
            return goal
        bound = goal.extra["bound"]
        new_value = np.clip(goal.value + rng.uniform(-bound * 0.2, bound * 0.2, size=3),
                            -bound * 0.5, bound * 0.5)
        return GoalState(value=new_value, extra=goal.extra)

    def coordination_quality(self, pos, vel, goal):
        if pos.shape[0] < 2:
            return 1.0
        var = float(pos.var(axis=0).mean())
        var0 = goal.extra["var0"] or 1.0
        return _clip01(1.0 - var / var0)


# --------------------------------------------------------------------------- #
# Tracking rendezvous — the imposed-task-timescale axis
# --------------------------------------------------------------------------- #
class TrackingRendezvousTask(CoordinationTask):
    """Gather on a goal that jumps on the τ clock and that only a few agents see.

    This is the task the ``delay/τ`` question needs, and the one the other four
    could not provide. ``rendezvous`` and ``consensus`` are τ-inert (their goal
    never reaches the dynamics or the metric); ``coverage_hold`` couples to τ but
    no primitive achieves it. Here:

    * ``perturb`` jumps the meeting point once per ``tau_sec`` — the **imposed**
      task timescale.
    * ``goal_accel`` pulls **only the informed fraction** toward it. Informed
      agents know the goal and their own true position, so that channel is not
      delayed.
    * ``Q`` scores *every* agent's distance to the goal, so the uninformed
      majority can only succeed by following the informed minority **through the
      delayed peer graph**.

    Coordination is therefore necessary (an uninformed agent cannot solve it
    alone), which is the same design principle that makes ``rendezvous`` a clean
    delay probe — but now with a *controllable* task timescale. A small τ (goal
    jumps often) plus a large delay (followers see stale leaders) is what should
    collapse Q, and whether it collapses on ``delay/τ`` is the open question.

    Setting ``informed_frac = 1.0`` makes the task individually solvable and Q
    should become delay-insensitive: a useful negative control, not a bug.
    """

    name = "tracking_rendezvous"

    def __init__(self, informed_frac: float = 0.15, length_scale_frac: float = 0.25,
                 jump_frac: float = 0.5):
        if not 0.0 < informed_frac <= 1.0:
            raise ValueError(f"informed_frac must be in (0, 1], got {informed_frac}")
        self.informed_frac = informed_frac
        self.length_scale_frac = length_scale_frac
        self.jump_frac = jump_frac

    def init_goal(self, rng, num_agents, bound):
        n_informed = max(1, int(round(self.informed_frac * num_agents)))
        informed = np.zeros(num_agents, dtype=bool)
        informed[rng.choice(num_agents, size=n_informed, replace=False)] = True
        value = rng.uniform(-bound * 0.5, bound * 0.5, size=3)
        return GoalState(value=value, extra={
            "bound": float(bound),
            "L0": float(bound) * self.length_scale_frac,
            "informed": informed,
        })

    def perturb(self, goal, ctx, rng):
        if not _reconfig_due(ctx):
            return goal
        bound = goal.extra["bound"]
        step = bound * self.jump_frac
        new_value = np.clip(goal.value + rng.uniform(-step, step, size=3),
                            -bound * 0.5, bound * 0.5)
        return GoalState(value=new_value, extra=goal.extra)

    def goal_accel(self, pos, vel, goal, max_accel):
        accel = np.zeros_like(pos)
        informed = goal.extra["informed"]
        if pos.shape[0] != informed.shape[0]:
            # Agent count changed under us (fault module); fall back to no pull
            # rather than mis-indexing a stale mask.
            return accel
        d = goal.value[None, :] - pos[informed]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        accel[informed] = max_accel * d / np.maximum(n, 1e-9)
        return accel

    def coordination_quality(self, pos, vel, goal):
        if pos.shape[0] == 0:
            return 0.0
        mean_dist = float(np.linalg.norm(pos - goal.value[None, :], axis=1).mean())
        L0 = goal.extra["L0"] or 1.0
        return _clip01(float(np.exp(-mean_dist / L0)))


ALL_TASKS = {
    "rendezvous": RendezvousTask,
    "tracking_rendezvous": TrackingRendezvousTask,
    "formation_hold": FormationHoldTask,
    "coverage_hold": CoverageHoldTask,
    "consensus": ConsensusTask,
}


__all__ = [
    "ALL_TASKS",
    "ConsensusTask",
    "CoverageHoldTask",
    "FormationHoldTask",
    "RendezvousTask",
    "TrackingRendezvousTask",
]
