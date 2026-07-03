"""Tests for gossamer.prediction — the P3 peer-state predictors.

Golden-by-construction where the physics allows (CV exactness), bracketed
consistency for the stochastic filter (Kalman NIS), and determinism throughout.
"""
import numpy as np
import pytest

from gossamer.prediction import (
    PREDICTORS,
    ConstantVelocityPredictor,
    KalmanPredictor,
    LinearPredictor,
    PeerHistory,
    STATE_DIM,
)

N = 20
DT = 0.5


def _cv_history(steps=6, seed=0):
    """A pure constant-velocity trajectory as a filled PeerHistory."""
    rng = np.random.default_rng(seed)
    pos0 = rng.uniform(-10, 10, size=(N, 3))
    vel = rng.uniform(-2, 2, size=(N, 3))
    hist = PeerHistory(N, capacity=steps)
    states = []
    for k in range(steps):
        pos = pos0 + vel * (k * DT)
        s = np.concatenate([pos, vel], axis=1)
        hist.push(s)
        states.append(s)
    return hist, pos0, vel, states


@pytest.mark.parametrize("name,cls", list(PREDICTORS.items()))
def test_zero_horizon_is_identity(name, cls):
    hist, *_ = _cv_history()
    pred = cls()
    pred.reset(N)
    out = pred.predict(hist, horizon_steps=0, dt=DT)
    assert out.shape == (N, STATE_DIM)
    # Position must equal the latest snapshot exactly at horizon 0.
    assert np.allclose(out[:, :3], hist.latest()[:, :3], atol=1e-9)


def test_constant_velocity_predictor_is_exact():
    hist, pos0, vel, _ = _cv_history()
    h = 5
    out = ConstantVelocityPredictor().predict(hist, horizon_steps=h, dt=DT)
    latest_t = (len(hist) - 1) * DT
    expected = pos0 + vel * (latest_t + h * DT)
    assert np.allclose(out[:, :3], expected, atol=1e-9)


def test_linear_predictor_exact_on_constant_velocity():
    hist, pos0, vel, _ = _cv_history()
    h = 4
    out = LinearPredictor(window=5).predict(hist, horizon_steps=h, dt=DT)
    latest_t = (len(hist) - 1) * DT
    expected = pos0 + vel * (latest_t + h * DT)
    assert np.allclose(out[:, :3], expected, atol=1e-6)


def test_kalman_converges_to_true_state_on_clean_cv():
    hist, pos0, vel, _ = _cv_history(steps=8)
    kf = KalmanPredictor(process_var=1e-4, meas_var=1e-3)
    kf.reset(N)
    out = kf.predict(hist, horizon_steps=3, dt=DT)
    latest_t = (len(hist) - 1) * DT
    expected = pos0 + vel * (latest_t + 3 * DT)
    # With near-perfect measurements the filter tracks the CV line closely.
    assert np.allclose(out[:, :3], expected, atol=1.0)


def test_kalman_nis_is_chi_square_consistent():
    """On noisy CV data the NIS should sit in a reasonable χ²(3) band (~3)."""
    rng = np.random.default_rng(3)
    pos0 = rng.uniform(-10, 10, size=(N, 3))
    vel = rng.uniform(-1, 1, size=(N, 3))
    meas_sd = 0.3
    hist = PeerHistory(N, capacity=10)
    for k in range(10):
        clean = pos0 + vel * (k * DT)
        noisy = clean + rng.normal(scale=meas_sd, size=(N, 3))
        hist.push(np.concatenate([noisy, np.tile(vel, 1)], axis=1))
    kf = KalmanPredictor(process_var=1e-3, meas_var=meas_sd ** 2)
    kf.reset(N)
    pred = kf.predict(hist, horizon_steps=1, dt=DT)
    actual_pos = pos0 + vel * (len(hist) * DT)
    actual = np.concatenate([actual_pos, vel], axis=1)
    cal = kf.calibration(pred, actual)
    assert "nis" in cal and "pos_rmse" in cal
    assert np.isfinite(cal["nis"])
    # Loose bracket: consistent filters land within an order of magnitude of 3.
    assert 0.1 < cal["nis"] < 60.0


@pytest.mark.parametrize("name,cls", list(PREDICTORS.items()))
def test_predictor_is_deterministic(name, cls):
    hist, *_ = _cv_history(seed=5)
    p1 = cls(); p1.reset(N)
    p2 = cls(); p2.reset(N)
    a = p1.predict(hist, horizon_steps=4, dt=DT)
    b = p2.predict(hist, horizon_steps=4, dt=DT)
    assert np.allclose(a, b)
