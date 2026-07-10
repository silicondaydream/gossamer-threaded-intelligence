"""
Unified coordination-task / quality API for the DCC trilogy.

Every benchmark task exposes a *normalized* quality signal so that every
coordination primitive is scored on one substrate:

* :meth:`CoordinationTask.init_goal` — build the task goal deterministically
  from a seed (the same deterministic-from-``rng`` idiom the benchmark
  :class:`~gossamer.benchmarks.scenarios.Scenario` uses).
* :meth:`CoordinationTask.perturb` — advance the goal on the *task-timescale*
  clock ``tau_sec`` (the moving-goal / reconfiguration rate). This is the τ
  half of the delay/τ ratio that the P1 phase diagram sweeps.
* :meth:`CoordinationTask.coordination_quality` — a scalar
  ``Q ∈ [0, 1]`` (higher is better), evaluated on the current ``(pos, vel)``
  against the current goal, so delay-vs-quality is measured identically for
  flocking, DMB, gossip-consensus, CRDT-intent and the market.

Unlike :class:`Scenario` (whose ``terminal_metric`` is an arbitrary-scale
"success number" bound to the leaderboard harness), the quality here is
normalized and evaluated *every step*. Tasks are deliberately decoupled from
the physics — they take plain ``np.ndarray`` state and are driven either by the
Leviathan engine (via the Maneuver.Map runner) or by an in-NumPy stepper in the
unit tests.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TaskContext:
    """Per-step runtime parameters handed to the task.

    ``tau_sec`` is the task-timescale: the characteristic time over which the
    goal moves / reconfigures. ``tau_sec <= 0`` (or ``None``) means a *static*
    goal — the ``delay/τ`` ratio is then zero regardless of delay.
    """
    step: int
    total_steps: int
    dt: float
    t: float
    tau_sec: Optional[float] = None


@dataclass(frozen=True)
class GoalState:
    """The task's current target, plus whatever bookkeeping ``perturb`` needs.

    Fields are all optional so one container serves every task:

    * ``point`` — a single (3,) rendezvous target.
    * ``offsets`` — (N, 3) per-agent formation offsets from the formation centre.
    * ``center`` — (3,) formation / coverage anchor.
    * ``target_cells`` — frozenset of (iy, ix) coverage cells to hold.
    * ``value`` — (k,) consensus target vector.
    * ``phase`` — a scalar the perturbation carries between steps (e.g. the
      current formation rotation angle or the OU state of a moving goal).
    * ``extra`` — task-specific scratch (never read by the runner).
    """
    point: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None
    center: Optional[np.ndarray] = None
    target_cells: Optional[frozenset] = None
    value: Optional[np.ndarray] = None
    phase: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


class CoordinationTask(ABC):
    """Base class for a DCC benchmark task with a normalized quality signal."""

    name: str = "base"

    @abstractmethod
    def init_goal(self, rng: np.random.Generator, num_agents: int, bound: float
                  ) -> GoalState:
        """Deterministically construct the initial goal from ``rng``."""
        ...

    @abstractmethod
    def perturb(self, goal: GoalState, ctx: TaskContext,
                rng: np.random.Generator) -> GoalState:
        """Advance the goal by one step on the ``tau_sec`` clock.

        Must return a *new* :class:`GoalState` (goals are immutable, like the
        CRDTs). A static task returns ``goal`` unchanged.
        """
        ...

    @abstractmethod
    def coordination_quality(self, pos: np.ndarray, vel: np.ndarray,
                             goal: GoalState) -> float:
        """Return coordination quality ``Q ∈ [0, 1]`` (higher is better)."""
        ...

    def goal_accel(self, pos: np.ndarray, vel: np.ndarray, goal: GoalState,
                   max_accel: float) -> np.ndarray:
        """Task objective as a per-agent acceleration toward the (moving) goal.

        The *objective* channel: an informed agent knows the goal and its own true
        position, so this term is **not** delayed. The coordination primitive
        (cohesion / spacing / consensus, on *delayed* peer state) is the channel
        delay degrades. Only when both are present does ``delay/τ`` become the
        controlling ratio.

        **This default returns a zero field, and a task that does not override it
        is τ-INERT** — the τ clock will move a goal that nothing reads and nothing
        steers toward, so ``tau_sec`` cannot affect ``Q``. That is the case for
        ``rendezvous`` and ``consensus``, which are *purely peer-derived by design*:
        their Q measures swarm compactness / variance reduction, which is exactly
        what makes coordination necessary and what the delay axis degrades. It is
        **not** an oversight, but it does mean:

        * sweeping ``tau_sec`` against those tasks yields a null by construction;
        * the "timescale" in the phase-diagram paper's claim is the swarm's
          *intrinsic* convergence time (set by the primitive's gain / radius /
          speed), not this configured ``tau_sec``.

        Use :class:`~gossamer.tasks.tasks.TrackingRendezvousTask` when you need an
        *imposed* task timescale that genuinely couples to Q.
        """
        return np.zeros_like(pos)


def _clip01(x: float) -> float:
    """Clamp a scalar into [0, 1] (guards against nan/inf blowing up a run)."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _reconfig_due(ctx: TaskContext) -> bool:
    """True on the step where the goal should jump, given ``tau_sec``.

    Fires roughly once per ``tau_sec`` of simulated time. Returns ``False`` for
    a static (non-positive / None) τ so callers can early-out.
    """
    tau = ctx.tau_sec
    if tau is None or tau <= 0.0 or ctx.dt <= 0.0:
        return False
    period_steps = max(1, int(round(tau / ctx.dt)))
    return ctx.step > 0 and (ctx.step % period_steps == 0)


__all__ = [
    "CoordinationTask",
    "GoalState",
    "TaskContext",
]
