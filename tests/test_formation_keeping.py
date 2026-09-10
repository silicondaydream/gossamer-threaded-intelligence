"""Separation-keeping policies (N4). The anchor is the CW closed-orbit condition."""

import math

import numpy as np
import pytest

from gossamer.algorithms.coordination.formation_keeping import (
    FORMATION_POLICIES,
    PolicyContext,
    conjunction_avoidance_policy,
    open_loop_policy,
    station_keeping_policy,
)

N = 0.0011          # mean motion, rad/s, roughly a 550 km orbit
CTX = PolicyContext(mean_motion=N, d_safe_m=50.0, control_period_s=600.0, dt_s=10.0)


def test_open_loop_never_commands_anything():
    pos = np.random.default_rng(0).normal(0, 300, (5, 3))
    vel = np.random.default_rng(1).normal(0, 0.1, (5, 3))
    assert np.array_equal(open_loop_policy(pos, vel, 0.0, CTX), np.zeros((5, 3)))


def test_station_keeping_nulls_the_secular_drift():
    """A CW orbit is closed iff vy = -2 n x, and drifts at -3(vy + 2 n x)
    otherwise. Applying the command must leave that term at zero."""
    pos = np.array([[300.0, 0.0, 0.0], [-150.0, 200.0, 50.0]])
    vel = np.array([[0.0, 0.1, 0.0], [0.0, 0.0, 0.0]])

    dv = station_keeping_policy(pos, vel, 0.0, CTX)
    after = vel + dv
    drift = -3.0 * (after[:, 1] + 2.0 * N * pos[:, 0])
    assert np.allclose(drift, 0.0, atol=1e-12)


def test_station_keeping_leaves_an_already_closed_orbit_alone():
    pos = np.array([[300.0, 0.0, 0.0]])
    vel = np.array([[0.0, -2.0 * N * 300.0, 0.0]])
    assert np.allclose(station_keeping_policy(pos, vel, 0.0, CTX), 0.0)


def test_a_policy_acts_on_the_control_period_not_on_every_sample():
    """Otherwise the Delta-v bill is a function of the sampling grid rather than
    of the controller."""
    pos = np.array([[300.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.1, 0.0]])

    assert np.any(station_keeping_policy(pos, vel, 0.0, CTX))
    assert np.any(station_keeping_policy(pos, vel, 600.0, CTX))
    assert not np.any(station_keeping_policy(pos, vel, 300.0, CTX))
    assert not np.any(station_keeping_policy(pos, vel, 10.0, CTX))


def test_the_control_cadence_is_the_same_wall_clock_at_two_sample_steps():
    pos = np.array([[300.0, 0.0, 0.0]])
    vel = np.array([[0.0, 0.1, 0.0]])
    fine = PolicyContext(mean_motion=N, d_safe_m=50.0, control_period_s=600.0, dt_s=1.0)
    coarse = PolicyContext(mean_motion=N, d_safe_m=50.0, control_period_s=600.0, dt_s=10.0)

    fired_fine = [t for t in range(0, 1200) if np.any(station_keeping_policy(pos, vel, float(t), fine))]
    fired_coarse = [t for t in range(0, 1200, 10)
                    if np.any(station_keeping_policy(pos, vel, float(t), coarse))]
    assert fired_fine == [0, 600]
    assert fired_coarse == [0, 600]


def test_conjunction_avoidance_pushes_a_close_pair_apart_symmetrically():
    pos = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    dv = conjunction_avoidance_policy(pos, vel, 0.0, CTX)

    assert np.allclose(dv[0], -dv[1])            # equal and opposite
    assert dv[1][0] > 0 and dv[0][0] < 0         # apart, not together
    assert np.allclose(dv.sum(axis=0), 0.0)      # no net momentum invented


def test_conjunction_avoidance_is_inert_outside_its_trigger():
    pos = np.array([[0.0, 0.0, 0.0], [5000.0, 0.0, 0.0]])
    vel = np.zeros((2, 3))
    assert np.allclose(conjunction_avoidance_policy(pos, vel, 0.0, CTX), 0.0)


def test_co_located_satellites_do_not_produce_a_nan_command():
    """Zero separation divides by zero on the way to a unit vector, and a NaN
    Delta-v propagates into the trajectory as a plausible number."""
    pos = np.zeros((2, 3))
    vel = np.zeros((2, 3))
    dv = conjunction_avoidance_policy(pos, vel, 0.0, CTX)
    assert np.all(np.isfinite(dv))
    assert np.any(dv)


def test_every_registered_policy_has_the_same_signature():
    pos = np.random.default_rng(2).normal(0, 300, (4, 3))
    vel = np.zeros((4, 3))
    for name, policy in FORMATION_POLICIES.items():
        out = np.asarray(policy(pos, vel, 0.0, CTX))
        assert out.shape == (4, 3), name
        assert np.all(np.isfinite(out)), name
