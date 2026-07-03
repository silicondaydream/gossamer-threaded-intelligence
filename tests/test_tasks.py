"""Tests for gossamer.tasks — the unified coordination-quality API.

Discipline (matching the rest of the suite): normalized-range laws, limit /
monotonicity brackets, and seed-determinism — not stored golden numbers.
"""
import numpy as np
import pytest

from gossamer.tasks import (
    ALL_TASKS,
    ConsensusTask,
    CoverageHoldTask,
    FormationHoldTask,
    RendezvousTask,
)
from gossamer.tasks.base import TaskContext


BOUND = 100.0
N = 64


def _rng(seed=0):
    return np.random.default_rng(seed)


@pytest.mark.parametrize("name,cls", list(ALL_TASKS.items()))
def test_quality_in_unit_interval_for_random_states(name, cls):
    task = cls()
    goal = task.init_goal(_rng(1), N, BOUND)
    rng = _rng(2)
    for _ in range(20):
        pos = rng.uniform(-BOUND, BOUND, size=(N, 3))
        vel = rng.normal(size=(N, 3))
        q = task.coordination_quality(pos, vel, goal)
        assert 0.0 <= q <= 1.0
        assert np.isfinite(q)


def test_rendezvous_quality_peaks_at_goal_and_monotone():
    task = RendezvousTask()
    goal = task.init_goal(_rng(3), N, BOUND)
    on_goal = np.tile(goal.point, (N, 1))
    vel = np.zeros((N, 3))
    q_on = task.coordination_quality(on_goal, vel, goal)
    assert q_on == pytest.approx(1.0, abs=1e-6)

    # Moving agents closer to the goal never decreases Q.
    far = on_goal + 30.0
    mid = on_goal + 10.0
    assert task.coordination_quality(mid, vel, goal) >= task.coordination_quality(far, vel, goal)

    # Scatter to the edges collapses Q toward 0.
    scatter = np.tile(goal.point, (N, 1)) + np.sign(np.arange(N * 3).reshape(N, 3) - 1) * BOUND
    assert task.coordination_quality(scatter, vel, goal) < 0.1


def test_formation_hold_quality_peaks_on_template():
    task = FormationHoldTask()
    goal = task.init_goal(_rng(4), N, BOUND)
    # Place agents exactly on offsets about an arbitrary centroid.
    center = np.array([5.0, -3.0, 2.0])
    pos = center[None, :] + goal.offsets
    vel = np.zeros((N, 3))
    assert task.coordination_quality(pos, vel, goal) == pytest.approx(1.0, abs=1e-6)
    # Perturbing agents off-template lowers Q.
    q_off = task.coordination_quality(pos + 20.0, vel, goal)
    assert q_off < 1.0


def test_coverage_hold_quality_bounds():
    task = CoverageHoldTask()
    goal = task.init_goal(_rng(5), 200, BOUND)
    rng = _rng(6)
    pos = rng.uniform(-BOUND, BOUND, size=(200, 3))
    vel = np.zeros((200, 3))
    q = task.coordination_quality(pos, vel, goal)
    assert 0.0 <= q <= 1.0
    # Empty swarm holds nothing.
    assert task.coordination_quality(np.empty((0, 3)), np.empty((0, 3)), goal) == 0.0


def test_consensus_quality_rises_as_swarm_converges():
    task = ConsensusTask()
    goal = task.init_goal(_rng(7), N, BOUND)
    rng = _rng(8)
    spread = rng.uniform(-BOUND * 0.25, BOUND * 0.25, size=(N, 3))
    tight = spread * 0.1
    vel = np.zeros((N, 3))
    assert task.coordination_quality(tight, vel, goal) >= task.coordination_quality(spread, vel, goal)


@pytest.mark.parametrize("name,cls", list(ALL_TASKS.items()))
def test_perturb_is_deterministic_under_seed(name, cls):
    task = cls()
    goal = task.init_goal(_rng(9), N, BOUND)
    ctx = TaskContext(step=10, total_steps=100, dt=1.0, t=10.0, tau_sec=5.0)
    g1 = task.perturb(goal, ctx, _rng(11))
    g2 = task.perturb(goal, ctx, _rng(11))
    # Compare whichever field the task uses.
    for attr in ("point", "offsets", "value"):
        a, b = getattr(g1, attr), getattr(g2, attr)
        if a is not None and b is not None:
            assert np.allclose(a, b)
    if g1.target_cells is not None:
        assert g1.target_cells == g2.target_cells


@pytest.mark.parametrize("name,cls", list(ALL_TASKS.items()))
def test_static_tau_leaves_goal_unchanged(name, cls):
    task = cls()
    goal = task.init_goal(_rng(12), N, BOUND)
    # tau_sec None / <=0 → static: perturb is a no-op every step.
    for tau in (None, 0.0, -1.0):
        ctx = TaskContext(step=50, total_steps=100, dt=1.0, t=50.0, tau_sec=tau)
        assert task.perturb(goal, ctx, _rng(13)) is goal
