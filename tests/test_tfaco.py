"""Tests for gossamer.algorithms.coordination.tfaco."""
import numpy as np
import pytest

from gossamer.algorithms.coordination.tfaco import (
    DensePheromoneGrid,
    PheromoneField,
    TFACOParams,
    tfaco_accel,
    tfaco_step,
)


def _params(**kw):
    base = dict(grid_w=16, grid_h=16, bound=100.0, evap_lambda=0.01, deposit_rate=1.0)
    base.update(kw)
    return TFACOParams(**base)


def test_deposit_then_decay():
    f = PheromoneField(_params(evap_lambda=0.01))
    f.deposit((3, 3), "A", t=0.0, amount=1.0)
    tau0 = f.tau((3, 3), 0.0)
    tau100 = f.tau((3, 3), 100.0)
    assert tau0 == 1.0
    assert tau100 < tau0
    assert tau100 == np.exp(-1.0)  # lambda * age = 0.01 * 100 = 1


def test_merge_is_convergent_and_idempotent():
    p = _params()
    a = PheromoneField(p).deposit((0, 0), "A", 0.0).deposit((1, 1), "A", 0.0)
    b = PheromoneField(p).deposit((2, 2), "B", 0.0).deposit((1, 1), "B", 0.0)
    ab = a.merge(b)
    ba = b.merge(a)
    t = 0.0
    assert np.allclose(ab.dense_grid(t), ba.dense_grid(t))   # commutative
    assert np.allclose(ab.merge(ab).dense_grid(t), ab.dense_grid(t))  # idempotent
    # All three deposited cells are present after merge.
    assert set(ab.deposits) == {(0, 0), (1, 1), (2, 2)}


def test_distinct_replicas_sum_same_replica_does_not_double_count():
    p = _params()
    a = PheromoneField(p).deposit((5, 5), "A", 0.0, amount=3.0)
    b = PheromoneField(p).deposit((5, 5), "B", 0.0, amount=5.0)
    merged = a.merge(b)
    assert merged.deposits[(5, 5)].value() == 8  # different replicas accumulate
    assert a.merge(a).deposits[(5, 5)].value() == 3  # idempotent, no double count


def test_last_t_takes_max_on_merge():
    p = _params()
    a = PheromoneField(p).deposit((1, 1), "A", t=5.0)
    b = PheromoneField(p).deposit((1, 1), "B", t=2.0)
    assert a.merge(b).last_t[(1, 1)] == 5.0


def test_coverage_increases_as_swarm_explores():
    p = _params(grid_w=32, grid_h=32, bound=100.0, heuristic_weight=0.2)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-20, 20, size=(20, 3))
    vel = np.zeros_like(pos)
    field = PheromoneField(p)
    cov0 = None
    for step in range(40):
        pos, vel, info = tfaco_step(pos, vel, field, t=float(step), dt=0.5,
                                    params=p, max_speed=5.0, max_accel=2.0)
        if cov0 is None:
            cov0 = info["coverage"]
    assert info["coverage"] > cov0  # exploration covers more cells over time
    assert np.all(np.linalg.norm(vel, axis=1) <= 5.0 + 1e-6)


def test_step_deposits_and_reports_pheromone():
    p = _params()
    pos = np.zeros((5, 3))
    vel = np.zeros((5, 3))
    field = PheromoneField(p)
    _, _, info = tfaco_step(pos, vel, field, t=0.0, dt=0.1, params=p)
    assert info["coverage"] > 0.0
    assert info["pheromone"].shape == (5,)
    assert np.all(info["pheromone"] > 0.0)  # everyone deposited on their cell


# --------------------------------------------------------------------------
# The dense-grid kernel — the TF-ACO the experiments actually run.
#
# Two implementations of one law is a liability unless something holds them to
# each other. `PheromoneField` is the CRDT (sparse, per-agent loop, proves
# convergence); `DensePheromoneGrid` is the vectorised twin that carries the
# numbers. The runner used to hold the fast one privately, and the two were kept
# in agreement by a docstring. These tests are what replace the docstring.
# --------------------------------------------------------------------------

def test_dense_grid_matches_the_crdt_field_on_one_replica():
    """The dense kernel must compute the SAME tau law as the CRDT it stands in for.

    This is the entire justification for having two implementations: on a single
    replica a CRDT reduces to its value, so the fast path is only legitimate if it
    lands on the same field. If this drifts, every TF-ACO number in the stack is
    produced by something with no proof behind it.
    """
    p = _params(grid_w=16, grid_h=16, deposit_rate=1.0)
    rng = np.random.default_rng(0)
    pos = rng.uniform(-90, 90, size=(25, 3))
    pos[:, 2] = 0.0

    grid = DensePheromoneGrid(p, t0=0.0)
    field = PheromoneField(p)

    t = 4.0
    # Dense path: one deposit+decay pass at time t.
    tfaco_accel(pos, np.zeros_like(pos), 0.1, t, grid)
    # CRDT path: the same deposits, one per agent, at the same time.
    for a in range(pos.shape[0]):
        field.deposit(field.cell_of(pos[a, :2]), a, t)

    assert np.allclose(grid.deposits, field.dense_grid(t), atol=1e-12)


def test_dense_grid_decays_with_the_same_exponential_law():
    p = _params(evap_lambda=0.5)
    grid = DensePheromoneGrid(p, t0=0.0)
    pos = np.zeros((1, 3))
    tfaco_accel(pos, np.zeros_like(pos), 0.1, 0.0, grid)

    cy, cx = np.nonzero(grid.deposits)
    deposited = float(grid.deposits[cy[0], cx[0]])
    # tau at a later time, with no further deposit, is the decayed mass.
    age = 3.0
    expected = deposited * np.exp(-p.evap_lambda * age)
    tau = grid.deposits * np.exp(
        -p.evap_lambda * np.maximum(0.0, age - grid.last_t)
    )
    assert tau[cy[0], cx[0]] == pytest.approx(expected)


def test_dense_grid_coverage_reads_the_grid_that_is_written():
    """`coverage()` must not be able to go hollow.

    The runner previously kept a live `PheromoneField` beside the dense kernel and
    asked IT for coverage — but the kernel deposits into the dense array and never
    touched the CRDT, so its deposits dict stayed empty and coverage() returned 0.0
    forever. A number that is always zero and never raises is the house failure
    mode: it reads as a measurement.
    """
    p = _params(grid_w=16, grid_h=16)
    grid = DensePheromoneGrid(p)
    assert grid.coverage() == 0.0

    rng = np.random.default_rng(1)
    pos = rng.uniform(-90, 90, size=(30, 3))
    pos[:, 2] = 0.0
    tfaco_accel(pos, np.zeros_like(pos), 0.1, 1.0, grid)
    assert grid.coverage() > 0.0


def test_dense_grid_steers_only_in_the_xy_plane():
    """z is left to the base flocking term; TF-ACO is a ground-plane policy."""
    p = _params()
    grid = DensePheromoneGrid(p)
    rng = np.random.default_rng(2)
    pos = rng.uniform(-90, 90, size=(10, 3))
    vel = rng.normal(0, 1, size=(10, 3))
    accel, _pher = tfaco_accel(pos, vel, 0.1, 0.0, grid, max_speed=1e9)
    assert np.allclose(accel[:, 2], 0.0)


def test_dense_grid_reports_per_agent_pheromone():
    p = _params()
    grid = DensePheromoneGrid(p)
    pos = np.zeros((5, 3))          # all agents in one cell -> they stack deposits
    _accel, pher = tfaco_accel(pos, np.zeros_like(pos), 0.1, 0.0, grid)
    assert pher.shape == (5,)
    assert np.all(pher > 0.0)
