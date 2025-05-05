import numpy as np
import pytest

from gossamer.environment.perception import (
    Observation,
    PerfectPerception,
    RangePerception,
    FieldOfViewPerception,
    DelayPerception,
    PacketLossPerception,
    BearingOnlyPerception,
    IntermittentBlindSpotPerception,
    StochasticDelayPerception,
)


def test_perfect_perception_no_noise():
    positions = np.array([[0, 0], [1, 1], [2, 2]])
    velocities = np.array([[0, 1], [1, 0], [-1, -1]])
    p = PerfectPerception()
    obs = p.perceive(0, positions, velocities)
    # Should perceive agents 1 and 2
    assert isinstance(obs, Observation)
    assert np.array_equal(obs.indices, np.array([1, 2]))
    assert np.array_equal(obs.positions, positions[[1, 2]])
    assert np.array_equal(obs.velocities, velocities[[1, 2]])


def test_perfect_perception_with_noise():
    # Use fixed seed for reproducibility
    np.random.seed(42)
    positions = np.zeros((3, 2))
    velocities = np.zeros((3, 2))
    p = PerfectPerception(noisy=True, pos_noise=1.0, vel_noise=1.0)
    obs = p.perceive(0, positions, velocities)
    # Noise should make obs.positions and obs.velocities non-zero
    assert obs.positions.shape == (2, 2)
    assert not np.allclose(obs.positions, 0)
    assert obs.velocities.shape == (2, 2)
    assert not np.allclose(obs.velocities, 0)


def test_range_perception_no_neighbors():
    positions = np.array([[0, 0], [5, 5], [10, 10]])
    velocities = np.zeros((3, 2))
    rp = RangePerception(radius=1.0)
    obs = rp.perceive(0, positions, velocities)
    assert isinstance(obs, Observation)
    # No neighbors within radius
    assert obs.indices.size == 0
    assert obs.positions.size == 0
    assert obs.velocities.size == 0


def test_range_perception_neighbors_and_noise():
    # seed for noise
    np.random.seed(0)
    positions = np.array([[0, 0], [1, 1], [3, 3]])
    velocities = np.array([[0, 0], [1, 1], [2, 2]])
    rp = RangePerception(radius=2.0, noisy=True, pos_noise=0.5, vel_noise=0.5)
    obs = rp.perceive(0, positions, velocities)
    # Only agent 1 is within sqrt(2) (~1.414) < 2
    assert np.array_equal(obs.indices, np.array([1]))
    # Check noise applied: values should differ from originals
    assert obs.positions.shape == (1, 2)
    assert not np.array_equal(obs.positions, positions[[1]])
    assert obs.velocities.shape == (1, 2)
    assert not np.array_equal(obs.velocities, velocities[[1]])
    
def test_field_of_view_perception():
    # Setup four agents in cardinal directions
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    velocities = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    # Narrow FOV: 90 degrees (pi/2), radius 2.0
    fovp = FieldOfViewPerception(radius=2.0, fov=np.pi/2)
    obs = fovp.perceive(0, positions, velocities)
    # Only the agent directly in front (index 1) should be seen
    assert np.array_equal(obs.indices, np.array([1]))
    # Wider FOV: 180 degrees (pi), include front and side
    fovp2 = FieldOfViewPerception(radius=2.0, fov=np.pi)
    obs2 = fovp2.perceive(0, positions, velocities)
    assert set(obs2.indices.tolist()) == {1, 2}

def test_delay_perception():
    # Two agents in 1D positions
    positions0 = np.array([[0.0], [1.0]])
    velocities0 = np.zeros((2, 1))
    positions1 = np.array([[10.0], [20.0]])
    velocities1 = np.zeros((2, 1))
    positions2 = np.array([[100.0], [200.0]])
    velocities2 = np.zeros((2, 1))
    base = PerfectPerception()
    dp = DelayPerception(base, delay_steps=1)
    # First perception uses initial state
    obs0 = dp.perceive(0, positions0, velocities0)
    assert np.array_equal(obs0.positions, np.array([[1.0]]))
    # Second call still sees positions0 due to delay
    obs1 = dp.perceive(0, positions1, velocities1)
    assert np.array_equal(obs1.positions, np.array([[1.0]]))
    # Third call sees positions1 (one-step delay)
    obs2 = dp.perceive(0, positions2, velocities2)
    assert np.array_equal(obs2.positions, np.array([[20.0]]))

def test_packet_loss_perception():
    positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    velocities = np.zeros((3, 2))
    base = PerfectPerception()
    # Full loss: no neighbors
    pl_all = PacketLossPerception(base, loss_prob=1.0)
    np.random.seed(0)
    obs_all = pl_all.perceive(0, positions, velocities)
    assert obs_all.indices.size == 0
    # No loss: all neighbors
    pl_none = PacketLossPerception(base, loss_prob=0.0)
    np.random.seed(0)
    obs_none = pl_none.perceive(0, positions, velocities)
    assert np.array_equal(obs_none.indices, np.array([1, 2]))
    # Partial loss: with fixed seed
    pl_half = PacketLossPerception(base, loss_prob=0.5)
    np.random.seed(1)
    obs_half = pl_half.perceive(0, positions, velocities)
    # Based on seed, only index 2 remains
    assert np.array_equal(obs_half.indices, np.array([2]))

def test_bearing_only_perception():
    # Agent at origin, heading along +x axis
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    velocities = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    base = PerfectPerception()
    bop = BearingOnlyPerception(base)
    obs = bop.perceive(0, positions, velocities)
    # indices should include 1 and 2
    assert obs.indices.tolist() == [1, 2]
    # bearings: neighbor 1 at 0 rad, neighbor 2 at pi/2 rad
    assert pytest.approx(obs.positions[0], rel=1e-6) == 0.0
    assert pytest.approx(obs.positions[1], rel=1e-6) == np.pi/2

def test_intermittent_blind_spot_perception():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    velocities = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    base = PerfectPerception()
    # No blind spot when inactive
    ibsp_off = IntermittentBlindSpotPerception(base, blind_width=2*np.pi, prob_active=0.0)
    np.random.seed(0)
    obs_off = ibsp_off.perceive(0, positions, velocities)
    assert np.array_equal(obs_off.indices, np.array([1, 2]))
    # Full blind spot when always active => drop all
    ibsp_on = IntermittentBlindSpotPerception(base, blind_width=2*np.pi, prob_active=1.0)
    np.random.seed(0)
    obs_on = ibsp_on.perceive(0, positions, velocities)
    assert obs_on.indices.size == 0

def test_stochastic_delay_perception():
    positions0 = np.array([[0.0], [1.0]])
    velocities0 = np.zeros((2, 1))
    positions1 = np.array([[10.0], [20.0]])
    velocities1 = np.zeros((2, 1))
    base = PerfectPerception()
    # Always delay by 1 step
    dp1 = StochasticDelayPerception(base, max_delay_steps=1, delay_sampler=lambda: 1)
    obs0 = dp1.perceive(0, positions0, velocities0)
    # first call, only one history => returns initial
    assert np.array_equal(obs0.positions, np.array([[1.0]]))
    obs1 = dp1.perceive(0, positions1, velocities1)
    # two histories, delay=1 => returns first
    assert np.array_equal(obs1.positions, np.array([[1.0]]))
    # No delay (immediate)
    dp0 = StochasticDelayPerception(base, max_delay_steps=1, delay_sampler=lambda: 0)
    obs2 = dp0.perceive(0, positions0, velocities0)
    assert np.array_equal(obs2.positions, np.array([[1.0]]))
    obs3 = dp0.perceive(0, positions1, velocities1)
    # delay=0 => returns latest
    assert np.array_equal(obs3.positions, np.array([[20.0]]))