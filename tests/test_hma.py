"""Tests for gossamer.algorithms.coordination.hma."""
import math

import numpy as np
import pytest

from gossamer.algorithms.coordination.hma import (
    DepotInventory,
    HMAParams,
    bid_utility,
    energy_aware_auction,
    erlang_c,
    mmc_metrics,
    soc_sigmoid,
)


def test_soc_sigmoid_shape():
    assert soc_sigmoid(0.30, s_crit=0.30, k=12.0) == pytest.approx(0.5)
    assert soc_sigmoid(0.9, 0.30, 12.0) > 0.99
    assert soc_sigmoid(0.05, 0.30, 12.0) < 0.05
    s = np.array([0.0, 0.2, 0.4, 0.6])
    assert np.all(np.diff(soc_sigmoid(s, 0.30, 12.0)) > 0)  # monotone increasing


def test_bid_utility_monotonicities():
    p = HMAParams()
    base = bid_utility(mass=100, e_travel=10, e_lift=5, soc=0.9, t_arrival=5, t_queue=0, params=p)
    more_mass = bid_utility(200, 10, 5, 0.9, 5, 0, p)
    more_energy = bid_utility(100, 40, 5, 0.9, 5, 0, p)
    low_soc = bid_utility(100, 10, 5, 0.1, 5, 0, p)
    assert more_mass > base       # more reward mass -> higher bid
    assert more_energy < base     # more energy cost -> lower bid
    assert low_soc < base         # depleted battery -> suppressed bid


def test_auction_prefers_nearest_when_one_slot():
    # Depot needs <= one hauler's capacity; nearest hauler should win it.
    hauler_pos = np.array([[1.0, 0.0], [50.0, 0.0]])
    soc = np.array([0.9, 0.9])
    depot_pos = np.array([[0.0, 0.0]])
    assign, cleared = energy_aware_auction(
        hauler_pos, soc, depot_pos, depot_mass=[300.0], hauler_capacity=500.0
    )
    assert assign[0] == 0
    assert assign[1] == -1
    assert cleared[0] == 300.0


def test_auction_zero_mass_depot_unassigned():
    hauler_pos = np.array([[0.0, 0.0]])
    assign, cleared = energy_aware_auction(
        hauler_pos, np.array([0.9]), np.array([[1.0, 0.0]]), depot_mass=[0.0]
    )
    assert assign[0] == -1
    assert cleared[0] == 0.0


def test_auction_multiple_haulers_cover_large_depot():
    hauler_pos = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    soc = np.full(3, 0.9)
    assign, cleared = energy_aware_auction(
        hauler_pos, soc, np.array([[0.0, 0.0]]), depot_mass=[1200.0],
        hauler_capacity=500.0,
    )
    assert np.count_nonzero(assign == 0) == 3   # 500 + 500 + 200
    assert cleared[0] == 1200.0


def test_auction_soc_deprioritizes_low_battery():
    # Two equidistant haulers, one slot; high-SOC hauler should win.
    hauler_pos = np.array([[1.0, 0.0], [-1.0, 0.0]])
    soc = np.array([0.05, 0.95])
    assign, _ = energy_aware_auction(
        hauler_pos, soc, np.array([[0.0, 0.0]]), depot_mass=[200.0],
        hauler_capacity=500.0,
    )
    assert assign[1] == 0   # high-SOC hauler wins
    assert assign[0] == -1


def test_auction_wear_breaks_ties():
    # Perfectly symmetric haulers; lower cumulative wear should win the slot.
    hauler_pos = np.array([[1.0, 0.0], [-1.0, 0.0]])
    soc = np.array([0.9, 0.9])
    wear = np.array([10.0, 0.0])
    assign, _ = energy_aware_auction(
        hauler_pos, soc, np.array([[0.0, 0.0]]), depot_mass=[200.0],
        hauler_capacity=500.0, hauler_wear=wear,
    )
    assert assign[1] == 0   # hauler 1 has lower wear
    assert assign[0] == -1


def test_depot_inventory_crdt_merge():
    a = DepotInventory().deposit(100, "m1").withdraw(30, "h1")
    b = DepotInventory().deposit(50, "m2")
    assert a.available() == 70
    merged = a.merge(b)
    # positives 100 (m1) + 50 (m2) = 150; negatives 30 (h1) -> 120.
    assert merged.available() == 120
    assert a.merge(a).available() == 70  # idempotent


def test_erlang_and_mmc_known_values():
    # M/M/1 with rho=0.5: Lq = rho^2/(1-rho) = 0.5.
    assert erlang_c(1, 0.5) == pytest.approx(0.5)
    m = mmc_metrics(lam=0.5, mu=1.0, c=1)
    assert m["rho"] == pytest.approx(0.5)
    assert m["Lq"] == pytest.approx(0.5)
    assert m["W"] == pytest.approx(2.0)
    # M/M/2 with offered load a=1.0: Erlang C ~ 0.3333.
    assert erlang_c(2, 1.0) == pytest.approx(1.0 / 3.0, abs=1e-6)


def test_mmc_overload_is_infinite():
    m = mmc_metrics(lam=2.0, mu=1.0, c=1)
    assert m["rho"] >= 1.0
    assert math.isinf(m["Lq"])
