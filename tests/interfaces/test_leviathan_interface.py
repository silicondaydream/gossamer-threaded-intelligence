import numpy as np
import pytest

from gossamer.interfaces.leviathan_interface import LeviathanInterface


class DummyEnv:
    def __init__(self):
        # Two agents in 2D
        self._pos = np.array([[0.0, 0.0], [1.0, 1.0]])
        self._vel = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.reset_called = False
        self.step_count = 0

    def reset(self):
        self.reset_called = True
        # Return fresh copies
        return {'positions': self._pos.copy(), 'velocities': self._vel.copy()}

    def step(self, actions=None):
        self.step_count += 1
        # Simple dynamics: pos += vel
        self._pos = self._pos + self._vel
        return {'positions': self._pos.copy(), 'velocities': self._vel.copy()}

    def compute_metrics(self):
        # Return step count as dummy metric
        return {'step_count': float(self.step_count)}


def test_init_calls_reset_and_sets_state():
    env = DummyEnv()
    adapter = LeviathanInterface(env)
    assert env.reset_called, "LeviathanInterface.__init__ should call env.reset()"
    # After reset, adapter should store initial positions and velocities
    np.testing.assert_array_equal(adapter.positions, np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_array_equal(adapter.velocities, np.array([[0.0, 1.0], [1.0, 0.0]]))

def test_step_updates_state_and_returns():
    env = DummyEnv()
    adapter = LeviathanInterface(env)
    pos1, vel1 = adapter.step()
    # After one step, positions should have incremented by initial velocities
    expected_pos = np.array([[0.0, 1.0], [2.0, 1.0]])
    np.testing.assert_array_equal(pos1, expected_pos)
    np.testing.assert_array_equal(vel1, np.array([[0.0, 1.0], [1.0, 0.0]]))
    # Internal state should also update
    np.testing.assert_array_equal(adapter.positions, expected_pos)
    np.testing.assert_array_equal(adapter.velocities, vel1)

def test_metrics_reflect_step_count():
    env = DummyEnv()
    adapter = LeviathanInterface(env)
    # Before any step
    m0 = adapter.metrics()
    assert m0['step_count'] == 0.0
    # After two steps
    adapter.step()
    adapter.step()
    m2 = adapter.metrics()
    assert m2['step_count'] == 2.0

def test_run_invokes_step_metrics_and_callback():
    env = DummyEnv()
    adapter = LeviathanInterface(env)
    calls = []
    live_match = []
    def callback(step, positions, velocities, metrics):
        # At call time the callback must receive the adapter's current state.
        live_match.append(np.array_equal(positions, adapter.positions))
        vel_copy = velocities.copy() if velocities is not None else None
        calls.append((step, positions.copy(), vel_copy, metrics.copy()))

    final_pos, final_vel = adapter.run(3, callback=callback)
    # Callback should be called for steps 0,1,2 with that step's live state.
    assert len(calls) == 3
    assert all(live_match)
    # DummyEnv dynamics are pos += vel each step from [[0,0],[1,1]].
    expected = [
        np.array([[0.0, 1.0], [2.0, 1.0]]),
        np.array([[0.0, 2.0], [3.0, 1.0]]),
        np.array([[0.0, 3.0], [4.0, 1.0]]),
    ]
    for idx, (step, pos, vel, met) in enumerate(calls):
        assert step == idx
        assert np.array_equal(pos, expected[idx])
        assert vel is None or isinstance(vel, np.ndarray)
        assert 'step_count' in met
    # Final returned positions equal adapter.positions
    np.testing.assert_array_equal(final_pos, adapter.positions)
    np.testing.assert_array_equal(final_vel, adapter.velocities)