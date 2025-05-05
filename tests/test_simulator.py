import numpy as np
import pytest

from gossamer.simulator import SwarmSimulator


def test_initialization_random():
    sim = SwarmSimulator(n_agents=5, n_dims=3)
    assert sim.positions.shape == (5, 3)
    assert sim.velocities.shape == (5, 3)


def test_invalid_positions():
    with pytest.raises(ValueError):
        SwarmSimulator(positions=[1, 2, 3])


def test_step_updates():
    sim = SwarmSimulator(n_agents=3, n_dims=2, dt=0.5)
    pos0 = sim.positions.copy()
    vel0 = sim.velocities.copy()
    pos1, vel1 = sim.step()
    assert pos1.shape == pos0.shape
    assert vel1.shape == vel0.shape
    # Positions should generally change
    assert not np.allclose(pos1, pos0)


def test_metrics_keys_and_types():
    sim = SwarmSimulator(n_agents=4, n_dims=2)
    m = sim.metrics()
    assert set(m.keys()) == {"cohesion", "alignment", "separation"}
    for v in m.values():
        assert isinstance(v, float)


def test_run_callback():
    sim = SwarmSimulator(n_agents=3, n_dims=2, dt=0.1)
    calls = []
    def cb(step, positions, velocities, metrics):
        calls.append((step, positions.copy(), velocities.copy(), metrics))
    sim.run(5, callback=cb)
    assert len(calls) == 5
    for i, (step, pos, vel, metrics) in enumerate(calls):
        assert step == i
        assert pos.shape == (3, 2)
        assert 'cohesion' in metrics


def test_velocity_mismatch():
    pos = np.zeros((3, 2))
    vel = np.zeros((4, 2))
    with pytest.raises(ValueError):
        SwarmSimulator(positions=pos, velocities=vel)