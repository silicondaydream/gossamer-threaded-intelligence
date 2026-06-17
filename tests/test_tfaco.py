"""Tests for gossamer.algorithms.coordination.tfaco."""
import numpy as np

from gossamer.algorithms.coordination.tfaco import (
    PheromoneField,
    TFACOParams,
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
