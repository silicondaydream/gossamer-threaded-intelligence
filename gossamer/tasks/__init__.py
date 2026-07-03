"""Unified coordination-task / quality API for the DCC trilogy (P1–P3).

Every task exposes a normalized ``coordination_quality(pos, vel, goal) -> Q``
in ``[0, 1]`` and a τ-paced ``perturb`` so ``delay/τ`` is a controlled axis.
"""
from gossamer.tasks.base import CoordinationTask, GoalState, TaskContext
from gossamer.tasks.tasks import (
    ALL_TASKS,
    ConsensusTask,
    CoverageHoldTask,
    FormationHoldTask,
    RendezvousTask,
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
]
