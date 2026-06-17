"""Tests for the OR-Tools MILP HMA baseline."""
import numpy as np
import pytest

pytest.importorskip("ortools")

from gossamer.algorithms.coordination.hma import HMAParams, bid_utility, energy_aware_auction
from gossamer.algorithms.coordination.milp_baseline import milp_assignment


def _total_utility(assignment, hauler_pos, soc, depot_pos, depot_mass, params, capacity):
    """Real-valued total bid utility of an assignment (for comparing schemes)."""
    total = 0.0
    for h, d in enumerate(assignment):
        if d < 0:
            continue
        load = min(capacity, depot_mass[d])
        dist = float(np.linalg.norm(hauler_pos[h] - depot_pos[d]))
        total += bid_utility(load, params.e_move * dist, params.e_lift * load,
                             soc[h], dist / params.speed, 0.0, params)
    return total


def test_milp_assignment_is_valid_and_respects_capacity():
    rng = np.random.default_rng(0)
    hauler_pos = rng.uniform(-10, 10, size=(6, 2))
    soc = np.full(6, 0.9)
    depot_pos = np.array([[0.0, 0.0], [8.0, 0.0]])
    depot_mass = [500.0, 1200.0]  # depot 0: 1 slot, depot 1: 2 slots (cap 500)
    assign, cleared = milp_assignment(hauler_pos, soc, depot_pos, depot_mass,
                                      hauler_capacity=500.0, time_budget_s=5.0)
    assert assign.shape == (6,)
    # at most one depot per hauler (trivially true: scalar per hauler)
    # depot slot limits respected
    assert np.count_nonzero(assign == 0) <= 1
    assert np.count_nonzero(assign == 1) <= 2
    # cleared mass never exceeds availability
    assert cleared[0] <= 500.0 and cleared[1] <= 1200.0


def test_milp_skips_zero_mass_depot():
    hauler_pos = np.array([[0.0, 0.0]])
    assign, cleared = milp_assignment(hauler_pos, np.array([0.9]),
                                      np.array([[1.0, 0.0]]), [0.0])
    assert assign[0] == -1
    assert cleared[0] == 0.0


def test_milp_total_utility_at_least_auction():
    # MILP is optimal within budget, so it must match or beat the greedy
    # auction on total bid utility (up to integer quantization).
    rng = np.random.default_rng(3)
    H = 8
    hauler_pos = rng.uniform(-20, 20, size=(H, 2))
    soc = rng.uniform(0.4, 1.0, size=H)
    depot_pos = np.array([[0.0, 0.0], [15.0, 5.0], [-10.0, -8.0]])
    depot_mass = [500.0, 500.0, 1000.0]
    params = HMAParams()
    cap = 500.0

    a_assign, _ = energy_aware_auction(hauler_pos, soc, depot_pos, depot_mass,
                                       params, hauler_capacity=cap, )
    m_assign, _ = milp_assignment(hauler_pos, soc, depot_pos, depot_mass, params,
                                  hauler_capacity=cap, time_budget_s=10.0)
    u_auction = _total_utility(a_assign, hauler_pos, soc, depot_pos, depot_mass, params, cap)
    u_milp = _total_utility(m_assign, hauler_pos, soc, depot_pos, depot_mass, params, cap)
    # Quantization tolerance: utility_scale=1000 over <= H assigned pairs.
    assert u_milp >= u_auction - (H / 1000.0) - 1e-6
