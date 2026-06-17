"""
Centralized MILP scheduler — the HMA paper's strong baseline (HMA §4.4).

A credible centralized comparator for the decentralized energy-aware auction in
:mod:`gossamer.algorithms.coordination.hma`: a CP-SAT assignment that *maximizes
the same bid utility* the auction optimizes locally, subject to one-depot-per-
hauler and depot capacity, under a wall-clock budget. Run head-to-head with
``energy_aware_auction`` it answers the reviewer's question — "how far is the
decentralized market from the centralized optimum?" — instead of the
hand-crippled "computationally feasible replanning cadence" strawman the earlier
draft used.

OR-Tools is an optional dependency, imported lazily so the rest of Gossamer
stays import-light. Utilities are scaled to integers (CP-SAT is integer-only);
the objective is therefore optimized up to the ``utility_scale`` quantization.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from gossamer.algorithms.coordination.hma import HMAParams, bid_utility


def milp_assignment(
    hauler_pos: np.ndarray,
    hauler_soc: np.ndarray,
    depot_pos: np.ndarray,
    depot_mass: Sequence[float],
    params: HMAParams = HMAParams(),
    *,
    hauler_capacity: float = 500.0,
    depot_queue_time: Optional[Sequence[float]] = None,
    time_budget_s: float = 60.0,
    utility_scale: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Centralized CP-SAT hauler→depot assignment maximizing total bid utility.

    Returns ``(assignment, cleared_mass)`` with the same shape/semantics as
    :func:`gossamer.algorithms.coordination.hma.energy_aware_auction`:
    ``assignment[h]`` is the depot hauler ``h`` won or ``-1``; a depot accepts
    at most ``floor(available_mass / capacity-per-hauler)`` haulers (so the
    cleared mass never exceeds availability), and only positive-utility pairs
    are eligible. Raises ``ImportError`` if OR-Tools is unavailable.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "milp_assignment requires OR-Tools. Install it with 'pip install ortools'."
        ) from e

    hauler_pos = np.asarray(hauler_pos, dtype=float)
    depot_pos = np.asarray(depot_pos, dtype=float)
    H, D = hauler_pos.shape[0], depot_pos.shape[0]
    soc = np.asarray(hauler_soc, dtype=float)
    qtime = np.zeros(D) if depot_queue_time is None else np.asarray(depot_queue_time, dtype=float)
    mass = np.array([float(m) for m in depot_mass], dtype=float)
    # Each hauler moves up to one capacity-load; a depot can serve this many.
    slots = np.floor(np.maximum(mass, 0.0) / max(hauler_capacity, 1e-9)).astype(int)

    model = cp_model.CpModel()
    x = {}
    util = {}
    for h in range(H):
        for d in range(D):
            load = min(hauler_capacity, mass[d])
            e_travel = params.e_move * float(np.linalg.norm(hauler_pos[h] - depot_pos[d]))
            e_lift = params.e_lift * load
            t_arrival = float(np.linalg.norm(hauler_pos[h] - depot_pos[d])) / max(params.speed, 1e-9)
            u = bid_utility(load, e_travel, e_lift, soc[h], t_arrival, qtime[d], params)
            if u > 0 and slots[d] > 0:
                x[h, d] = model.NewBoolVar(f"x_{h}_{d}")
                util[h, d] = int(round(u * utility_scale))

    # Each hauler assigned to at most one depot.
    for h in range(H):
        vars_h = [x[h, d] for d in range(D) if (h, d) in x]
        if vars_h:
            model.Add(sum(vars_h) <= 1)
    # Each depot serves at most its slot count.
    for d in range(D):
        vars_d = [x[h, d] for h in range(H) if (h, d) in x]
        if vars_d:
            model.Add(sum(vars_d) <= int(slots[d]))

    model.Maximize(sum(util[h, d] * x[h, d] for (h, d) in x))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_budget_s)
    solver.Solve(model)

    assignment = -np.ones(H, dtype=int)
    cleared = np.zeros(D, dtype=float)
    for (h, d), var in x.items():
        if solver.Value(var) == 1:
            assignment[h] = d
            cleared[d] += min(hauler_capacity, mass[d])
    return assignment, cleared


__all__ = ["milp_assignment"]
