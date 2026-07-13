"""The classical comparators, and the properties that make them fair baselines.

These arrived here by migration out of `maneuver-map/backend/app/policies.py`,
where they were real algorithm code sitting in the orchestration layer. Moving the
code without moving the coverage would have been half a migration — the whole
argument for the move is that an algorithm outside Gossamer is untested by
Gossamer, unshipped in the wheel the papers pin, and unrunnable by anyone
reproducing a result from the public package.

The tests worth having here are not "does it return an array". They are the
properties that make a baseline *honest*, because a comparator that is quietly
crippled turns a paper's headline into an artifact of its denominator:

  * Lévy must be reproducible under its seed (it is pure noise; unseeded, every
    number it produces is unfalsifiable) and must genuinely NOT communicate.
  * Greedy must actually be greedy — nearest unvisited — and must record what it
    covered, since coverage is the axis DMB claims to beat it on.
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.baselines import (
    CoverageGrid, LevyState, greedy_coverage_accel, levy_flight_accel,
)


def _swarm(n=40, seed=3):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-40, 40, size=(n, 3))
    pos[:, 2] = 0.0
    vel = rng.normal(0, 1.0, size=(n, 3))
    vel[:, 2] = 0.0
    return pos, vel


# --------------------------------------------------------------------------
# Levy
# --------------------------------------------------------------------------
def test_levy_is_reproducible_under_its_seed():
    """Same seed, same draws. The walk is pure noise; the seed is its only anchor."""
    _pos, vel = _swarm()
    a1 = levy_flight_accel(vel, 0.1, LevyState.create(vel.shape[0], seed=7))
    a2 = levy_flight_accel(vel, 0.1, LevyState.create(vel.shape[0], seed=7))
    assert np.array_equal(a1, a2)


def test_levy_differs_across_seeds():
    """The reproducibility test above must not be passing because it is degenerate."""
    _pos, vel = _swarm()
    a1 = levy_flight_accel(vel, 0.1, LevyState.create(vel.shape[0], seed=7))
    a2 = levy_flight_accel(vel, 0.1, LevyState.create(vel.shape[0], seed=8))
    assert not np.allclose(a1, a2)


def test_levy_does_not_communicate():
    """Moving a peer must not change an agent's action.

    This is THE property that makes Lévy the "structure-free" reference: if the
    baseline could see its neighbours, DMB's advantage over it would be measuring
    something other than structure. It takes no position argument at all, which is
    the strongest possible form of this guarantee — this test pins that signature
    against a future "helpful" refactor that hands it `pos`.
    """
    _pos, vel = _swarm()
    st = LevyState.create(vel.shape[0], seed=11)
    a_ref = levy_flight_accel(vel, 0.1, st)

    moved = vel.copy()
    moved[0] += 50.0                      # violently perturb agent 0's state
    st2 = LevyState.create(vel.shape[0], seed=11)
    a_moved = levy_flight_accel(moved, 0.1, st2)

    # Every agent EXCEPT the perturbed one is untouched by its neighbour's change.
    assert np.allclose(a_ref[1:], a_moved[1:])


def test_levy_step_speed_is_bounded_by_max_speed():
    _pos, vel = _swarm()
    st = LevyState.create(vel.shape[0], seed=5)
    accel = levy_flight_accel(vel, 0.1, st, max_speed=3.0)
    new_vel = vel + accel * 0.1
    # Only reorienting agents get a redrawn (bounded) velocity; the rest coast.
    reoriented = ~np.isclose(accel, 0.0).all(axis=1)
    assert np.all(np.linalg.norm(new_vel[reoriented], axis=1) <= 3.0 + 1e-6)


def test_levy_reorientation_countdown_advances():
    """Agents that just reoriented must not reorient again on the very next step."""
    _pos, vel = _swarm()
    st = LevyState.create(vel.shape[0], seed=2)
    levy_flight_accel(vel, 0.1, st)       # step 1: everyone is due (countdown 0)
    assert np.all(st.countdown > -1.0)    # countdowns were redrawn, then decremented
    a2 = levy_flight_accel(vel, 0.1, st)  # step 2: most agents should now coast
    coasting = np.isclose(a2, 0.0).all(axis=1)
    assert coasting.any(), "no agent coasted — the countdown is not being honoured"


# --------------------------------------------------------------------------
# Greedy coverage
# --------------------------------------------------------------------------
def test_greedy_marks_the_cells_it_visits():
    pos, vel = _swarm()
    grid = CoverageGrid.create(resolution=32, bound=50.0)
    greedy_coverage_accel(pos, vel, 0.1, grid)
    assert grid.visited.sum() > 0
    assert 0.0 < grid.coverage() <= 1.0


def test_greedy_steers_toward_an_unvisited_cell():
    """A lone agent in a fresh grid must accelerate, not sit still.

    The failure this guards is the silent one: if the window search found nothing
    (an off-by-one, an inverted mask), the agent would coast, the run would look
    fine, and greedy would score a flat zero on coverage — handing DMB a win it
    did not earn.
    """
    pos = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    grid = CoverageGrid.create(resolution=32, bound=50.0)
    accel = greedy_coverage_accel(pos, vel, 0.1, grid)
    assert np.linalg.norm(accel) > 0.0


def test_greedy_coasts_when_its_window_is_fully_visited():
    """Saturated window -> coast. This is the documented escape hatch, so pin it."""
    pos = np.zeros((1, 3))
    vel = np.array([[1.0, 0.0, 0.0]])
    grid = CoverageGrid.create(resolution=32, bound=50.0)
    grid.visited[:] = True                # everything already covered
    accel = greedy_coverage_accel(pos, vel, 0.1, grid)
    assert np.allclose(accel, 0.0), "a saturated window must coast, not steer"


def test_greedy_has_no_pheromone_memory():
    """Revisiting a cell must not decay it back to 'unvisited'.

    Greedy is what stigmergy is worth *over*. If its visitation grid decayed, it
    would quietly become a crude TF-ACO and the comparison would lose its meaning.
    """
    pos = np.zeros((1, 3))
    vel = np.zeros((1, 3))
    grid = CoverageGrid.create(resolution=32, bound=50.0)
    greedy_coverage_accel(pos, vel, 0.1, grid)
    before = grid.visited.copy()
    for _ in range(20):
        greedy_coverage_accel(pos, vel, 0.1, grid)
    assert np.all(grid.visited[before]), "a visited cell became unvisited again"
