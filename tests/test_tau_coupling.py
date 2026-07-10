"""Which tasks does `tau_sec` actually control?

`perturb` moving the goal is NOT sufficient. The goal must also reach the
dynamics (`goal_accel`) or the metric (`coordination_quality`). Four of the five
tasks fail one of those, which is why sweeping `tau_sec` against the phase-diagram
grid would have returned a null *by construction* — the same defect class as the
degenerate P2 bandwidth frontier.

These tests pin the coupling table in `gossamer.tasks.tasks`'s docstring so it
cannot rot, and prove `TrackingRendezvousTask` is genuinely τ-coupled.
"""
import numpy as np
import pytest

from gossamer.tasks.base import CoordinationTask, TaskContext, _reconfig_due
from gossamer.tasks.tasks import ALL_TASKS, TrackingRendezvousTask


def _ctx(step, tau, dt=1.0):
    return TaskContext(step=step, total_steps=1000, dt=dt, t=step * dt, tau_sec=tau)


# --------------------------------------------------------------------------
# The τ clock itself
# --------------------------------------------------------------------------

def test_reconfig_clock_fires_once_per_tau():
    fires = [i for i in range(101) if _reconfig_due(_ctx(i, tau=20.0))]
    assert fires == [20, 40, 60, 80, 100]


@pytest.mark.parametrize("tau", [None, 0.0, -1.0])
def test_static_tau_never_fires(tau):
    assert not any(_reconfig_due(_ctx(i, tau)) for i in range(50))


# --------------------------------------------------------------------------
# The coupling table
# --------------------------------------------------------------------------

TAU_INERT = ("rendezvous", "consensus")
TAU_COUPLED = ("tracking_rendezvous",)


@pytest.mark.parametrize("name", TAU_INERT)
def test_peer_derived_tasks_are_tau_inert_by_design(name):
    """Their Q is swarm compactness / variance — it cannot see the goal.

    This is deliberate (it is what makes coordination *necessary*), but it means
    `tau_sec` is not the timescale in "delay / coordination timescale".
    """
    task = ALL_TASKS[name]()
    rng = np.random.default_rng(0)
    goal = task.init_goal(rng, 20, 100.0)
    pos = rng.uniform(-50, 50, (20, 3))
    vel = np.zeros((20, 3))

    # Drive the goal far away; Q must not move.
    moved = goal
    for step in (10, 20, 30):
        moved = task.perturb(moved, _ctx(step, tau=10.0), rng)
    assert task.coordination_quality(pos, vel, goal) == pytest.approx(
        task.coordination_quality(pos, vel, moved))

    # ...and the objective channel is a zero field.
    assert np.array_equal(task.goal_accel(pos, vel, moved, 1.0), np.zeros_like(pos))


@pytest.mark.parametrize("name", TAU_COUPLED)
def test_tau_coupled_tasks_read_the_moving_goal(name):
    task = ALL_TASKS[name]()
    rng = np.random.default_rng(0)
    goal = task.init_goal(rng, 20, 100.0)
    pos = rng.uniform(-50, 50, (20, 3))
    vel = np.zeros((20, 3))

    moved = task.perturb(goal, _ctx(10, tau=10.0), rng)
    assert not np.array_equal(moved.value, goal.value), "goal must move on the clock"
    assert task.coordination_quality(pos, vel, goal) != pytest.approx(
        task.coordination_quality(pos, vel, moved)), "Q must see the goal"
    assert np.any(task.goal_accel(pos, vel, moved, 1.0) != 0.0), "goal must pull"


def test_base_goal_accel_is_a_zero_field():
    """Documented explicitly: a task that does not override it is τ-inert."""
    class Bare(CoordinationTask):
        name = "bare"
        def init_goal(self, rng, n, bound): return None
        def perturb(self, goal, ctx, rng): return goal
        def coordination_quality(self, pos, vel, goal): return 0.0

    pos = np.ones((5, 3))
    assert np.array_equal(Bare().goal_accel(pos, pos, None, 1.0), np.zeros((5, 3)))


# --------------------------------------------------------------------------
# TrackingRendezvousTask
# --------------------------------------------------------------------------

def test_only_the_informed_fraction_is_pulled_toward_the_goal():
    task = TrackingRendezvousTask(informed_frac=0.2)
    rng = np.random.default_rng(1)
    goal = task.init_goal(rng, 50, 100.0)
    pos = rng.uniform(-50, 50, (50, 3))

    accel = task.goal_accel(pos, np.zeros_like(pos), goal, max_accel=1.0)
    pulled = np.linalg.norm(accel, axis=1) > 0
    assert pulled.sum() == 10                       # 20% of 50
    assert np.array_equal(pulled, goal.extra["informed"])
    # Uninformed agents get nothing — they must follow through the peer graph.
    assert np.allclose(accel[~goal.extra["informed"]], 0.0)


def test_informed_agents_are_pulled_toward_the_goal_not_away():
    task = TrackingRendezvousTask(informed_frac=1.0)
    rng = np.random.default_rng(2)
    goal = task.init_goal(rng, 8, 100.0)
    pos = rng.uniform(-50, 50, (8, 3))
    accel = task.goal_accel(pos, np.zeros_like(pos), goal, max_accel=1.0)

    to_goal = goal.value[None, :] - pos
    to_goal /= np.linalg.norm(to_goal, axis=1, keepdims=True)
    assert np.allclose(accel, to_goal)              # unit pull, magnitude max_accel


def test_quality_scores_every_agent_not_just_the_informed():
    """Otherwise the uninformed majority could be abandoned and Q would still be 1."""
    task = TrackingRendezvousTask(informed_frac=0.5)
    rng = np.random.default_rng(3)
    goal = task.init_goal(rng, 10, 100.0)
    informed = goal.extra["informed"]

    pos = np.tile(goal.value, (10, 1)).astype(float)
    q_all_on_goal = task.coordination_quality(pos, np.zeros_like(pos), goal)
    pos[~informed] += 500.0                          # strand the followers
    q_followers_lost = task.coordination_quality(pos, np.zeros_like(pos), goal)

    assert q_all_on_goal > 0.99
    assert q_followers_lost < q_all_on_goal


def test_goal_stays_inside_the_domain_under_repeated_jumps():
    task = TrackingRendezvousTask()
    rng = np.random.default_rng(4)
    goal = task.init_goal(rng, 10, 100.0)
    for step in range(1, 500):
        goal = task.perturb(goal, _ctx(step, tau=5.0), rng)
    assert np.all(np.abs(goal.value) <= 50.0 + 1e-9)  # clipped to bound*0.5


def test_goal_is_static_without_a_tau():
    task = TrackingRendezvousTask()
    rng = np.random.default_rng(5)
    goal = task.init_goal(rng, 10, 100.0)
    for step in range(1, 100):
        assert task.perturb(goal, _ctx(step, tau=None), rng) is goal


def test_informed_mask_survives_a_perturb():
    task = TrackingRendezvousTask(informed_frac=0.3)
    rng = np.random.default_rng(6)
    goal = task.init_goal(rng, 20, 100.0)
    moved = task.perturb(goal, _ctx(10, tau=10.0), rng)
    assert np.array_equal(moved.extra["informed"], goal.extra["informed"])


def test_goal_accel_tolerates_an_agent_count_change():
    """The fault module can drop agents mid-run; a stale mask must not mis-index."""
    task = TrackingRendezvousTask()
    rng = np.random.default_rng(7)
    goal = task.init_goal(rng, 20, 100.0)
    smaller = np.zeros((12, 3))
    assert np.array_equal(task.goal_accel(smaller, smaller, goal, 1.0), np.zeros((12, 3)))


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_informed_frac_raises(bad):
    with pytest.raises(ValueError, match="informed_frac"):
        TrackingRendezvousTask(informed_frac=bad)


def test_registered_in_all_tasks():
    assert ALL_TASKS["tracking_rendezvous"] is TrackingRendezvousTask
