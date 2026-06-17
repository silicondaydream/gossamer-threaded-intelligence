"""
Golden-value tests for gossamer.metrics.

These pin the criticality and information-theoretic estimators to known
closed-form answers and known qualitative behaviours. The DMB + TF-ACO paper's
headline results (universality class, critical exponents, transfer-entropy
information flow) are only as trustworthy as these functions, so they get
exact-value coverage, not just smoke tests.
"""
import numpy as np
import pytest

from gossamer.metrics.criticality import (
    avalanche_size_distribution,
    binder_cumulant,
    branching_ratio,
    correlation_length,
    susceptibility,
    velocity_correlation,
)
from gossamer.metrics.info import (
    mutual_information_histogram,
    mutual_information_ksg,
    transfer_entropy,
)


# ---- criticality: closed forms ----

def test_susceptibility_equals_N_times_variance():
    series = [0.1, 0.5, 0.9]
    # var(ddof=1) = 0.16; N=10 -> 1.6
    assert susceptibility(series, n_agents=10) == pytest.approx(1.6)


def test_binder_cumulant_ordered_limit_is_two_thirds():
    # A constant (fully ordered) series gives U = 1 - 1/3 = 2/3.
    assert binder_cumulant([0.5] * 50) == pytest.approx(2.0 / 3.0)


def test_binder_cumulant_disordered_limit_near_zero():
    # A zero-mean Gaussian has <x^4> = 3<x^2>^2, so U -> 0 (disordered phase).
    rng = np.random.default_rng(0)
    x = rng.normal(size=200_000)
    assert binder_cumulant(x) == pytest.approx(0.0, abs=0.02)


def test_branching_ratio_known_value():
    # mean([2,4]) / mean([1,2]) = 3 / 1.5 = 2.
    assert branching_ratio([1, 2, 4]) == pytest.approx(2.0)


def test_avalanche_size_distribution():
    vals, counts = avalanche_size_distribution([0, 1, 2, 0, 3, 0])
    # two avalanches: 1+2=3 and 3 -> size 3 occurs twice.
    assert list(vals) == [3]
    assert list(counts) == [2]


def test_correlation_length_finite_when_field_anticorrelates_at_range():
    rng = np.random.default_rng(1)
    # Two clusters with opposite velocity fluctuations.
    a = rng.normal(scale=0.3, size=(40, 3)) + np.array([0.0, 0.0, 0.0])
    b = rng.normal(scale=0.3, size=(40, 3)) + np.array([10.0, 0.0, 0.0])
    pos = np.vstack([a, b])
    vel = np.vstack([
        np.tile([1.0, 0.0, 0.0], (40, 1)),
        np.tile([-1.0, 0.0, 0.0], (40, 1)),
    ])
    r_edges, c = velocity_correlation(vel, pos, n_bins=24)
    xi = correlation_length(r_edges, c)
    assert np.isfinite(xi)
    assert 0.0 < xi < r_edges[-1]


def test_correlation_length_infinite_under_global_order():
    # Identical velocities -> zero fluctuation -> never crosses zero -> inf.
    rng = np.random.default_rng(2)
    pos = rng.normal(size=(30, 3))
    vel = np.tile([1.0, 0.0, 0.0], (30, 1))
    r_edges, c = velocity_correlation(vel, pos, n_bins=16)
    assert correlation_length(r_edges, c) == float("inf")


# ---- information theory: known behaviours ----

def test_histogram_mi_zero_for_independent_positive_for_dependent():
    rng = np.random.default_rng(3)
    x = rng.uniform(size=4000)
    y_indep = rng.uniform(size=4000)
    mi_indep = mutual_information_histogram(x, y_indep, bins=16)
    mi_dep = mutual_information_histogram(x, x, bins=16)
    assert abs(mi_indep) < 0.05
    assert mi_dep > 1.0
    assert mi_dep > mi_indep


def test_ksg_mi_zero_for_independent_positive_for_dependent():
    rng = np.random.default_rng(4)
    x = rng.normal(size=800)
    y_indep = rng.normal(size=800)
    y_dep = x + rng.normal(scale=0.1, size=800)
    assert abs(mutual_information_ksg(x, y_indep, k=4)) < 0.1
    assert mutual_information_ksg(x, y_dep, k=4) > 0.3


def test_transfer_entropy_is_directional():
    # target's future is the source's past: s drives t, not vice versa.
    rng = np.random.default_rng(5)
    n = 1500
    s = rng.normal(size=n)
    t = np.empty(n)
    t[0] = 0.0
    t[1:] = s[:-1] + rng.normal(scale=1e-2, size=n - 1)
    te_forward = transfer_entropy(s, t, lag=1, k=4)
    te_backward = transfer_entropy(t, s, lag=1, k=4)
    assert te_forward > te_backward
    assert te_forward > 0.1
