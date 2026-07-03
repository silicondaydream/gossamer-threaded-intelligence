"""Tests for gossamer.algorithms.coordination.primitive — CoordinationPrimitive.

Locks the wrappers to (a) the acceleration/telemetry contract, (b) equivalence
with the un-wrapped algorithms, and (c) seed-determinism.
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.algorithms.coordination.primitive import (
    PRIMITIVES,
    DMBPrimitive,
    FlockingPrimitive,
    GossipConsensusPrimitive,
    NoCommReference,
)
from gossamer.tasks.base import TaskContext


N = 48
DT = 0.5


def _state(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-20, 20, size=(N, 3))
    vel = rng.normal(size=(N, 3))
    return pos, vel


def _ctx():
    return TaskContext(step=0, total_steps=10, dt=DT, t=0.0, tau_sec=None)


@pytest.mark.parametrize("name,factory", list(PRIMITIVES.items()))
def test_act_returns_accel_shape_and_finite(name, factory):
    prim = factory()
    pos, vel = _state()
    prim.reset(N, np.random.default_rng(0))
    accel, telem = prim.act(pos, vel, DT, _ctx(), {})
    assert accel.shape == (N, 3)
    assert np.all(np.isfinite(accel))
    assert isinstance(telem, dict)


def test_flocking_primitive_matches_flock_step():
    """Golden-by-equivalence: the wrapper reproduces the un-wrapped algorithm."""
    pos, vel = _state(1)
    params = dict(alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5,
                  neighbor_radius=10.0, separation_distance=1.0, max_speed=5.0)
    _, new_vel = flock_step(pos, vel, DT, **params)
    expected_accel = (new_vel - vel) / DT

    accel, _ = FlockingPrimitive().act(pos, vel, DT, _ctx(), params)
    assert np.allclose(accel, expected_accel)


def test_dmb_primitive_is_deterministic_under_seed():
    pos, vel = _state(2)
    params = {"neighbor_radius": 12.0, "sensing_noise": 0.5, "max_speed": 5.0}

    p1 = DMBPrimitive(); p1.reset(N, np.random.default_rng(7))
    a1, _ = p1.act(pos, vel, DT, _ctx(), params)
    p2 = DMBPrimitive(); p2.reset(N, np.random.default_rng(7))
    a2, _ = p2.act(pos, vel, DT, _ctx(), params)
    assert np.allclose(a1, a2)


def test_no_comm_reference_only_damps_velocity():
    pos, vel = _state(3)
    accel, _ = NoCommReference().act(pos, vel, DT, _ctx(), {"damping": 0.1})
    assert np.allclose(accel, -0.1 * vel)
    assert not NoCommReference.uses_comm


def test_gossip_moves_agents_toward_neighborhood_mean():
    # Two tight clusters far apart: gossip should pull each agent toward its
    # local mean, shrinking within-cluster spread.
    rng = np.random.default_rng(4)
    a = rng.normal(scale=1.0, size=(N // 2, 3)) + np.array([-15.0, 0, 0])
    b = rng.normal(scale=1.0, size=(N // 2, 3)) + np.array([15.0, 0, 0])
    pos = np.vstack([a, b])
    vel = np.zeros_like(pos)
    accel, _ = GossipConsensusPrimitive().act(pos, vel, DT, _ctx(),
                                              {"neighbor_radius": 8.0, "consensus_gain": 1.0})
    new_vel = vel + accel * DT
    new_pos = pos + new_vel * DT
    # Per-cluster spread should not increase.
    def spread(p):
        return float(p.var(axis=0).sum())
    assert spread(new_pos[:N // 2]) <= spread(pos[:N // 2]) + 1e-6
    assert spread(new_pos[N // 2:]) <= spread(pos[N // 2:]) + 1e-6
