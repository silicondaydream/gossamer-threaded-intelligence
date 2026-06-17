"""Tests for gossamer.algorithms.coordination.dmb."""
import numpy as np
import pytest

from gossamer.algorithms.coordination.dmb import (
    DMBParams,
    density_modulated_weights,
    dmb_step,
    local_density,
    weighted_boids_update,
)


def test_weights_at_inflection_are_midpoints():
    p = DMBParams(d_crit=12.0, w_align_max=1.0, w_coh_max=1.0,
                  w_sep_min=1.0, w_sep_max=3.0)
    wa, wc, ws = density_modulated_weights(np.array([12.0]), p)
    assert wa[0] == pytest.approx(0.5)
    assert wc[0] == pytest.approx(0.5)
    assert ws[0] == pytest.approx(2.0)  # midpoint of [1, 3]


def test_weights_limits_and_monotonicity():
    p = DMBParams(d_crit=12.0, k_align=0.8, k_coh=0.8, k_sep=0.8,
                  w_align_max=1.0, w_coh_max=1.0, w_sep_min=1.0, w_sep_max=3.0)
    d = np.array([0.0, 6.0, 12.0, 18.0, 100.0])
    wa, wc, ws = density_modulated_weights(d, p)
    # Alignment and cohesion decrease with density; separation increases.
    assert np.all(np.diff(wa) <= 1e-12)
    assert np.all(np.diff(wc) <= 1e-12)
    assert np.all(np.diff(ws) >= -1e-12)
    # Limits.
    assert wa[0] == pytest.approx(p.w_align_max, abs=2e-3)
    assert wa[-1] == pytest.approx(0.0, abs=2e-3)
    assert ws[-1] == pytest.approx(p.w_sep_max, abs=2e-3)
    assert ws[0] == pytest.approx(p.w_sep_min, abs=2e-3)


def test_flat_schedules_recover_constant_weights():
    p = DMBParams(k_align=0.0, k_coh=0.0, k_sep=0.0,
                  w_align_max=1.0, w_coh_max=1.0, w_sep_min=1.0, w_sep_max=3.0)
    wa, wc, ws = density_modulated_weights(np.array([0.0, 5.0, 50.0]), p)
    assert np.allclose(wa, 0.5)
    assert np.allclose(wc, 0.5)
    assert np.allclose(ws, 2.0)  # min + (max-min)*0.5


def test_local_density_counts_neighbors():
    # Two agents 1.0 apart, a third far away.
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    dens = local_density(pos, radius=2.0)
    assert dens[0] == 1
    assert dens[1] == 1
    assert dens[2] == 0


def test_dmb_step_shapes_and_speed_clamp():
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=10.0, size=(50, 3))
    vel = rng.normal(scale=1.0, size=(50, 3))
    new_pos, new_vel, info = dmb_step(pos, vel, dt=0.1, max_speed=5.0,
                                      neighbor_radius=8.0)
    assert new_pos.shape == pos.shape
    assert new_vel.shape == vel.shape
    speeds = np.linalg.norm(new_vel, axis=1)
    assert np.all(speeds <= 5.0 + 1e-6)
    assert set(info) == {"density", "w_align", "w_coh", "w_sep"}
    assert info["density"].shape == (50,)


def test_dmb_step_is_deterministic_under_seed():
    pos = np.random.default_rng(1).normal(scale=10.0, size=(30, 3))
    vel = np.zeros_like(pos)
    a = dmb_step(pos, vel, 0.1, sensing_noise=0.5, rng=np.random.default_rng(7))
    b = dmb_step(pos, vel, 0.1, sensing_noise=0.5, rng=np.random.default_rng(7))
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[1], b[1])


def test_separation_pushes_crowded_agents_apart():
    # Two agents well inside separation distance, no initial velocity, strong
    # separation schedule -> they should move apart over a step.
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    p = DMBParams(d_crit=0.0, w_sep_min=2.0, w_sep_max=3.0, density_radius=2.0)
    new_pos, _, _ = dmb_step(pos, vel, dt=0.1, params=p,
                             neighbor_radius=2.0, separation_distance=1.0,
                             max_speed=5.0)
    assert np.linalg.norm(new_pos[0] - new_pos[1]) > np.linalg.norm(pos[0] - pos[1])


def test_max_accel_clamp_respected():
    rng = np.random.default_rng(2)
    pos = rng.normal(scale=5.0, size=(40, 3))
    vel = rng.normal(scale=1.0, size=(40, 3))
    dt, max_accel = 0.1, 2.0
    _, new_vel, _ = dmb_step(pos, vel, dt, neighbor_radius=6.0,
                             max_accel=max_accel, max_speed=1e9)
    dv = np.linalg.norm(new_vel - vel, axis=1)
    assert np.all(dv <= max_accel * dt + 1e-6)


def test_weighted_boids_update_scalar_weights_separate_crowded():
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    new_pos, _ = weighted_boids_update(pos, vel, dt=0.1, w_align=0.0, w_coh=0.0,
                                       w_sep=3.0, neighbor_radius=2.0,
                                       separation_distance=1.0, max_speed=5.0)
    assert np.linalg.norm(new_pos[0] - new_pos[1]) > np.linalg.norm(pos[0] - pos[1])


def test_weighted_boids_update_per_agent_weights_honored():
    # Only agent 0 has separation weight; agent 1 has none -> agent 0 moves more.
    pos = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]])
    vel = np.zeros_like(pos)
    w_sep = np.array([5.0, 0.0])
    new_pos, _ = weighted_boids_update(pos, vel, dt=0.1, w_align=0.0, w_coh=0.0,
                                       w_sep=w_sep, neighbor_radius=2.0,
                                       separation_distance=1.0, max_speed=5.0)
    moved0 = np.linalg.norm(new_pos[0] - pos[0])
    moved1 = np.linalg.norm(new_pos[1] - pos[1])
    assert moved0 > moved1
    assert moved1 == pytest.approx(0.0, abs=1e-9)
