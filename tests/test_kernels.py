"""The vectorised kernels are the only implementation; the per-agent loops are
the oracle that proves them right.

Before this, `maneuver-map/backend/app/policies.py` carried a private fork of
these kernels and the equivalence was asserted only in a docstring ("the
vectorised twin of gossamer weighted_boids_update"). It happened to be true — but
nothing checked it, and Orrery is about to add three more primitives.
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.dmb import DMBParams, dmb_step
from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.algorithms.coordination.kernels import (
    as_col, boids_accel_edges, kdtree_edges,
)
from gossamer.algorithms.coordination.primitive import DMBPrimitive, FlockingPrimitive

BOIDS = dict(alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5,
             neighbor_radius=5.0, separation_distance=1.0, max_speed=5.0)


def _state(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-10, 10, (n, 3)), rng.uniform(-1, 1, (n, 3))


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_kernel_matches_the_per_agent_loop_reference(seed):
    """`boids_accel_edges` must agree with `flock_step` to floating-point noise.

    They differ only in reduction order (`np.add.at` over an edge list vs a
    per-agent Python sum), so the tolerance is fp, not algorithmic.
    """
    pos, vel = _state(seed=seed)
    dt = 0.1
    _, new_vel = flock_step(pos, vel, dt, **BOIDS)
    accel_loop = (new_vel - vel) / dt

    eu, ev = kdtree_edges(pos, BOIDS["neighbor_radius"])
    accel_kernel = boids_accel_edges(
        pos, vel, eu, ev, dt, BOIDS["alignment_weight"], BOIDS["cohesion_weight"],
        BOIDS["separation_weight"], BOIDS["separation_distance"], BOIDS["max_speed"])

    assert np.allclose(accel_loop, accel_kernel, atol=1e-12)


def test_flocking_primitive_uses_the_kernel_and_still_matches_the_oracle():
    pos, vel = _state()
    dt = 0.1
    accel, telem = FlockingPrimitive().act(pos, vel, dt, None, BOIDS)
    _, new_vel = flock_step(pos, vel, dt, **BOIDS)
    assert np.allclose(accel, (new_vel - vel) / dt, atol=1e-12)
    assert telem == {}


def test_dmb_primitive_density_is_the_radius_graph_degree():
    pos, vel = _state()
    p = DMBPrimitive()
    p.reset(pos.shape[0], np.random.default_rng(0))
    _, telem = p.act(pos, vel, 0.1, None, {"neighbor_radius": 5.0})

    eu, ev = kdtree_edges(pos, 5.0)
    expected = np.zeros(pos.shape[0])
    np.add.at(expected, eu, 1.0)
    np.add.at(expected, ev, 1.0)
    assert np.array_equal(telem["density"], expected)


def test_dmb_primitive_is_deterministic_under_sensing_noise():
    pos, vel = _state()
    params = {"neighbor_radius": 5.0, "sensing_noise": 0.5}

    def once():
        p = DMBPrimitive()
        p.reset(pos.shape[0], np.random.default_rng(11))
        return p.act(pos, vel, 0.1, None, params)[0]

    assert np.allclose(once(), once())


def test_sensing_noise_perturbs_the_view_not_the_truth():
    """Noise must change the decision, not the integrated state (the caller owns pos)."""
    pos, vel = _state()
    before = pos.copy()
    p = DMBPrimitive()
    p.reset(pos.shape[0], np.random.default_rng(3))
    p.act(pos, vel, 0.1, None, {"neighbor_radius": 5.0, "sensing_noise": 1.0})
    assert np.array_equal(pos, before)


def test_kdtree_edges_are_upper_triangular_and_within_radius():
    pos, _ = _state(n=60, seed=5)
    eu, ev = kdtree_edges(pos, 4.0)
    assert np.all(eu < ev)
    d = np.linalg.norm(pos[eu] - pos[ev], axis=1)
    assert np.all(d <= 4.0 + 1e-12)


def test_kdtree_edges_handles_degenerate_sizes():
    for n in (0, 1):
        eu, ev = kdtree_edges(np.zeros((n, 3)), 1.0)
        assert eu.size == 0 and ev.size == 0


def test_boids_accel_with_no_edges_is_zero_for_isolated_agents():
    pos = np.array([[0.0, 0, 0], [1000.0, 0, 0]])
    vel = np.zeros((2, 3))
    eu, ev = kdtree_edges(pos, 1.0)
    accel = boids_accel_edges(pos, vel, eu, ev, 0.1, 1.0, 1.0, 1.5, 1.0, 5.0)
    assert np.allclose(accel, 0.0)


def test_as_col_broadcasts_scalars_and_per_agent_weights():
    assert as_col(2.0) == 2.0
    assert as_col(np.arange(4.0)).shape == (4, 1)


def test_dmb_step_loop_reference_still_exists_for_oracle_use():
    """The loop version is retained deliberately; deleting it removes the oracle."""
    pos, vel = _state()
    out = dmb_step(pos, vel, 0.1, DMBParams(), neighbor_radius=5.0)
    assert len(out) == 3
