"""Unified coordination-task / quality API for the DCC trilogy (P1–P3).

Every task exposes a normalized ``coordination_quality(pos, vel, goal) -> Q`` in
``[0, 1]`` and a τ-paced ``perturb``. Note that ``perturb`` moving the goal is
NOT sufficient for ``tau_sec`` to control Q — the goal must also reach the
dynamics (``goal_accel``) or the metric. See the table in ``gossamer.tasks.tasks``;
``TrackingRendezvousTask`` is the task with a genuinely controlled τ axis.
"""
from gossamer.tasks.base import CoordinationTask, GoalState, TaskContext
from gossamer.tasks.tasks import (
    ALL_TASKS,
    ConsensusTask,
    CoverageHoldTask,
    FormationHoldTask,
    RendezvousTask,
    TrackingRendezvousTask,
)

__all__ = [
    "ALL_TASKS",
    "ConsensusTask",
    "CoordinationTask",
    "CoverageHoldTask",
    "FormationHoldTask",
    "GoalState",
    "RendezvousTask",
    "TaskContext",
    "TrackingRendezvousTask",
]
