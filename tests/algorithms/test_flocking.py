import numpy as np
import pytest

from gossamer.algorithms.coordination.flocking import flock_step


def test_two_agent_cohesion():
    # Two agents should move toward each other under cohesion
    positions = np.array([[0.0, 0.0], [10.0, 0.0]])
    velocities = np.zeros((2, 2))
    # Enable only cohesion
    new_pos, new_vel = flock_step(
        positions,
        velocities,
        dt=1.0,
        alignment_weight=0.0,
        cohesion_weight=1.0,
        separation_weight=0.0,
        neighbor_radius=20.0,
        separation_distance=1.0,
        max_speed=100.0,
    )
    # Agent 0 should move positive x; agent 1 should move negative x
    assert new_pos[0, 0] > 0
    assert new_pos[1, 0] < 10
    assert new_vel[0, 0] > 0
    assert new_vel[1, 0] < 0


def test_shape_consistency():
    # Output shapes should match input
    n, dim = 5, 3
    positions = np.random.rand(n, dim)
    velocities = np.random.rand(n, dim)
    new_pos, new_vel = flock_step(positions, velocities, dt=0.5)
    assert new_pos.shape == positions.shape
    assert new_vel.shape == velocities.shape


def test_invalid_shapes_raise():
    # Mismatched shapes should raise
    positions = np.zeros((3, 2))
    velocities = np.zeros((4, 2))
    with pytest.raises(ValueError):
        flock_step(positions, velocities, dt=0.1)