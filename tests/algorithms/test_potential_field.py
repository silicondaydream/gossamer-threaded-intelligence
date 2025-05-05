import numpy as np
import pytest

from gossamer.algorithms.navigation.potential_field import potential_field_force, potential_field_step


def test_potential_field_attraction_only():
    pos = np.array([0.0, 0.0])
    goal = np.array([1.0, 0.0])
    force = potential_field_force(pos, goal, obstacles=None, k_att=2.0, k_rep=0.0)
    # Attractive force = -k_att * (pos - goal) = [2, 0]
    assert np.allclose(force, np.array([2.0, 0.0]))


def test_potential_field_repulsion_blocks_obstacle():
    pos = np.array([1.0, 0.0])
    goal = np.array([0.0, 0.0])
    obstacle = np.array([[1.5, 0.0]])  # obstacle ahead
    # Repulsion dominates attraction
    force = potential_field_force(pos, goal, obstacles=obstacle, k_att=1.0, k_rep=100.0, rep_range=1.0)
    # force x-component should be negative (pushed back)
    assert force[0] < 0


def test_potential_field_step_moves_toward_goal():
    pos = np.array([0.0, 0.0])
    goal = np.array([0.0, 1.0])
    new_pos = potential_field_step(pos, goal, obstacles=None, step_size=0.5, k_att=1.0)
    # New position should have y > 0 and x == 0
    assert np.isclose(new_pos[0], 0.0)
    assert new_pos[1] > 0

def test_potential_field_invalid_shapes():
    with pytest.raises(ValueError):
        potential_field_force([0, 0, 0], [1, 1], obstacles=None)
    with pytest.raises(ValueError):
        potential_field_force([0, 0], [1, 1, 1], obstacles=None)