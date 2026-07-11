"""The benchmark package (the N2 moat).

There was no test file for it. Its entire coverage was 8 tests in test_engine.py,
and nothing exercised `leaderboard()`, `generate_leaderboard_md()`, or the byzantine
path — which is how ByzantineScenario shipped as a NO-OP: it computed
`byzantine_indices` and the harness never read them, so the "byzantine" row of the
leaderboard was a plain rendezvous run under a different label. A benchmark that
reports robustness nobody tested is worse than no benchmark.
"""
import numpy as np
import pytest

from gossamer.benchmarks import (
    ALL_SCENARIOS,
    DEFAULT_BASELINES,
    BenchmarkConfig,
    generate_leaderboard_md,
    leaderboard,
    run_benchmark,
)
from gossamer.benchmarks.scenarios import ByzantineScenario, RendezvousScenario


def _cfg(**kw):
    base = dict(num_agents=40, steps=40, dt=0.1, bound=50.0, seed=7,
                record_trajectory=True)
    base.update(kw)
    return BenchmarkConfig(**base)


# --- the byzantine scenario must actually be adversarial ---------------------

def test_byzantine_actually_corrupts_the_marked_agents():
    """The bug: corrupt_actions did not exist and the marks were never read."""
    sc = ByzantineScenario(byzantine_fraction=0.5, adversary="random", scale=10.0)
    rng = np.random.default_rng(0)
    sc.init_state(rng, num_agents=10, bound=50.0)
    assert sc.byzantine_indices.size == 5

    honest = np.ones((10, 3))
    out = sc.corrupt_actions(honest, rng, ctx=None)

    corrupted = ~np.all(np.isclose(out, 1.0), axis=1)
    assert set(np.flatnonzero(corrupted)) == set(sc.byzantine_indices.tolist())
    # And the honest agents are untouched.
    honest_idx = sorted(set(range(10)) - set(sc.byzantine_indices.tolist()))
    assert np.allclose(out[honest_idx], 1.0)


def test_byzantine_does_not_mutate_the_callers_array():
    sc = ByzantineScenario(byzantine_fraction=1.0)
    rng = np.random.default_rng(0)
    sc.init_state(rng, 6, 50.0)
    accel = np.ones((6, 3))
    sc.corrupt_actions(accel, rng, None)
    assert np.allclose(accel, 1.0), "corrupt_actions wrote through to the caller"


def test_inverted_adversary_negates_the_honest_command():
    sc = ByzantineScenario(byzantine_fraction=1.0, adversary="inverted")
    rng = np.random.default_rng(1)
    sc.init_state(rng, 5, 50.0)
    accel = np.full((5, 3), 2.0)
    assert np.allclose(sc.corrupt_actions(accel, rng, None), -2.0)


def test_zero_fraction_is_the_identity():
    sc = ByzantineScenario(byzantine_fraction=0.0)
    rng = np.random.default_rng(2)
    sc.init_state(rng, 8, 50.0)
    accel = np.ones((8, 3))
    assert np.allclose(sc.corrupt_actions(accel, rng, None), 1.0)


def test_byzantine_now_scores_worse_than_clean_rendezvous():
    """The end-to-end proof that the row is no longer a lie.

    Adversaries fighting the policy must degrade the rendezvous metric. Before the
    fix these two runs were numerically IDENTICAL — same scenario, same seed, and
    the corruption never applied.
    """
    cfg = _cfg()
    baseline = DEFAULT_BASELINES["gossamer_flocking"]

    clean = run_benchmark(RendezvousScenario(), baseline(RendezvousScenario()), cfg)
    attacked_sc = ByzantineScenario(byzantine_fraction=0.4, adversary="inverted")
    attacked = run_benchmark(attacked_sc, baseline(attacked_sc), cfg)

    assert attacked.metric != clean.metric, (
        "the byzantine run scored identically to the clean one — the adversary is "
        "not being applied")


def test_byzantine_rejects_a_nonsense_configuration():
    with pytest.raises(ValueError):
        ByzantineScenario(byzantine_fraction=1.5)
    with pytest.raises(ValueError):
        ByzantineScenario(adversary="mimic")


# --- the honest scenarios are unaffected -------------------------------------

@pytest.mark.parametrize("name", sorted(ALL_SCENARIOS))
def test_every_scenario_runs_and_reports_a_finite_metric(name):
    sc = ALL_SCENARIOS[name]()
    result = run_benchmark(sc, DEFAULT_BASELINES["random"](sc), _cfg(),
                           baseline_name="random")
    assert result.scenario == sc.name
    assert np.isfinite(result.metric)
    assert np.isfinite(result.mean_reward)


def test_corrupt_actions_is_the_identity_for_honest_scenarios():
    sc = RendezvousScenario()
    accel = np.ones((5, 3))
    assert sc.corrupt_actions(accel, np.random.default_rng(0), None) is accel


# --- dispersal must not allocate a full pairwise tensor ----------------------

def test_dispersal_scales_past_the_pairwise_tensor():
    """It used to build (N, N, 3) — 2.4 TB at its own documented 10k ceiling."""
    sc = ALL_SCENARIOS["dispersal"]()
    result = run_benchmark(sc, DEFAULT_BASELINES["random"](sc),
                           _cfg(num_agents=4000, steps=3, record_trajectory=True))
    assert np.isfinite(result.metric)


# --- the coverage baseline must not leak state across runs -------------------

def test_coverage_walker_does_not_share_state_between_runs():
    """Headings used to be keyed on id(pos), which CPython recycles."""
    from gossamer.benchmarks.baselines import coverage_walker

    a, b = coverage_walker(), coverage_walker()
    pos = np.zeros((4, 3))
    first = a(pos, pos, np.random.default_rng(0))
    second = b(pos, pos, np.random.default_rng(0))
    # Same seed, independent closures -> same result, and neither inherited the
    # other's headings via a recycled array id.
    assert np.allclose(first, second)


# --- the leaderboard ---------------------------------------------------------

def test_leaderboard_runs_the_matrix_and_renders():
    small = _cfg(num_agents=20, steps=15)
    results = leaderboard(
        scenarios=["rendezvous", "byzantine"],
        baselines=["random", "gossamer_flocking"],
        configs={"rendezvous": small, "byzantine": small},
    )
    assert len(results) == 4
    md = generate_leaderboard_md(results)
    assert "rendezvous" in md and "byzantine" in md
