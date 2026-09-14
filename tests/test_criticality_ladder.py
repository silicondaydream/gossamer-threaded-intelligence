"""Tests for the criticality ladder (roadmap 18, the DMB absorption).

The reductions are tested against SYNTHETIC ladders with a known answer, not
against a run of the engine: a crossing-finder that agrees with whatever the
engine produced has verified nothing. The engine is exercised separately, and
only for the properties that must hold at any configuration.
"""
from __future__ import annotations

import numpy as np
import pytest

from gossamer.benchmarks.criticality import (
    DRIFT_TOLERANCE, CriticalitySpec, audit_ladder, binder_crossing,
    reduce_ladder, run_cell, susceptibility_peaks)
from gossamer.benchmarks.harness import (
    BenchmarkConfig, ScenarioConfigConflictError, run_benchmark)
from gossamer.benchmarks.baselines import vicsek_alignment
from gossamer.benchmarks.scenarios import ALL_SCENARIOS, VicsekTransitionScenario
from gossamer.engine import ReferenceEngine


# --------------------------------------------------------------------------
# the ladder must not densify, and must not be reachable any other way
# --------------------------------------------------------------------------

def test_resizing_the_scenario_holds_the_validated_density():
    """A finite-size ladder at a fixed box measures DENSITY, not size, and a
    scaling argument built on it is about the wrong axis. The engine's own
    scalability sweep follows the same rule for the same reason."""
    base = VicsekTransitionScenario()
    rho = (base.required_config["num_agents"]
           / (2.0 * base.required_config["bound"]) ** 3)
    for n in (50, 100, 200, 400, 800, 1600):
        rc = VicsekTransitionScenario(num_agents=n).required_config
        got = rc["num_agents"] / (2.0 * rc["bound"]) ** 3
        assert rc["num_agents"] == n
        assert got == pytest.approx(rho, rel=1e-12), n


def test_the_default_scenario_is_byte_identical_to_the_published_row():
    """Adding the ladder must not move what N2 published."""
    assert VicsekTransitionScenario().required_config == {"num_agents": 200,
                                                          "bound": 22.0}
    assert ALL_SCENARIOS["vicsek_transition"]().required_config == {
        "num_agents": 200, "bound": 22.0}


def test_resizing_through_the_config_is_still_refused():
    """The only supported way to move N is the scenario constructor, which
    rescales the box with it. Setting `num_agents` on the config instead is the
    exact route that once left this row 100x too sparse, and it must stay shut."""
    with pytest.raises(ScenarioConfigConflictError):
        run_benchmark(VicsekTransitionScenario(), vicsek_alignment(),
                      BenchmarkConfig(num_agents=64, steps=2),
                      engine=ReferenceEngine())


def test_a_degenerate_size_is_refused():
    with pytest.raises(ValueError):
        VicsekTransitionScenario(num_agents=1)


# --------------------------------------------------------------------------
# the observer hook is exactly neutral
# --------------------------------------------------------------------------

def test_the_observer_does_not_move_the_result():
    """It follows the OrbitalMarketWorld observer's discipline: off by default on
    the same branch as absent, no randomness drawn, read-only. If it moved a
    number, every leaderboard row would silently depend on whether anyone was
    watching."""
    cfg = BenchmarkConfig(num_agents=40, steps=25, seed=5)
    scenario = VicsekTransitionScenario(noise_eta=3.0, num_agents=40)
    without = run_benchmark(scenario, vicsek_alignment(), cfg,
                            engine=ReferenceEngine())
    seen = []
    with_obs = run_benchmark(scenario, vicsek_alignment(), cfg,
                             engine=ReferenceEngine(),
                             observer=lambda s, p, v: seen.append(s))
    assert without.metric == with_obs.metric
    assert without.mean_reward == with_obs.mean_reward
    assert seen == list(range(cfg.steps))


def test_a_run_with_no_trajectory_is_refused_not_scored_as_zero():
    """`terminal_metric` reduces the trajectory and returns 0.0 for an empty one,
    so this combination produced a publishable-looking total coordination failure
    with nothing raised. Nothing shipped sets it, which is why it survived."""
    with pytest.raises(ValueError, match="record_trajectory"):
        run_benchmark(VicsekTransitionScenario(), vicsek_alignment(),
                      BenchmarkConfig(steps=4, record_trajectory=False),
                      engine=ReferenceEngine())


# --------------------------------------------------------------------------
# the reductions, against ladders whose answer is known
# --------------------------------------------------------------------------

def _synthetic(eta_c=10.0, sizes=(100, 200, 400), etas=np.arange(6.0, 15.0, 1.0),
               crossing=True):
    """A ladder with a Binder crossing planted at ``eta_c``.

    U4 runs from 2/3 (ordered) to a lower disordered value, with the steepness
    growing with N so the curves cross at eta_c. With ``crossing=False`` every
    size shares one curve, which is what a crossover looks like.
    """
    rows = []
    for n in sizes:
        # `crossing=False` gives every size the SAME curve, which is the
        # size-independence a crossover shows -- not merely a weaker N-dependence.
        k = 0.6 * ((n / sizes[0]) ** 0.5 if crossing else 1.0)
        for eta in etas:
            u4 = 0.30 + (2.0 / 3.0 - 0.30) / (1.0 + np.exp(k * (eta - eta_c)))
            psi = 1.0 / (1.0 + np.exp(1.2 * (eta - eta_c)))
            chi = n * 0.02 * np.exp(-((eta - eta_c) ** 2) / 2.0)
            rows.append({"n_agents": n, "eta": float(eta), "n_seeds": 20,
                         "psi": float(psi), "psi_sd": 0.01,
                         "chi": float(chi), "chi_between_seed": float(chi * 3),
                         "u4": float(u4), "u4_sd": 0.001,
                         "max_abs_drift": 0.001})
    return rows


def test_binder_crossing_recovers_a_planted_crossing():
    out = binder_crossing(_synthetic(eta_c=10.0))
    assert out["n_pairs_with_crossing"] == 2
    assert out["eta_c_mean"] == pytest.approx(10.0, abs=0.15)
    assert out["eta_c_spread"] < 0.2


def test_binder_crossing_reports_none_rather_than_the_nearest_grid_point():
    """A grid that does not contain the crossing must say so. Snapping to the
    nearest point manufactures an eta_c out of a sweep that never saw one."""
    rows = _synthetic(eta_c=10.0, etas=np.arange(12.0, 15.0, 0.5))
    out = binder_crossing(rows)
    assert out["n_pairs_with_crossing"] == 0
    assert out["eta_c_mean"] is None


def test_a_crossover_produces_no_crossing():
    """The negative result must be reachable, or the finder always finds one.

    Coincident curves are the trap: the difference is identically zero, so the
    naive `d[i] == 0 => crossing` reads the leftmost grid point as eta_c and
    reports a critical point for the clearest possible crossover.
    """
    out = binder_crossing(_synthetic(crossing=False))
    assert out["n_pairs_with_crossing"] == 0
    assert out["n_pairs_coincident"] == len(out["pairs"])
    assert out["eta_c_mean"] is None


def test_the_pairwise_spread_is_reported_not_averaged_away():
    """Finite-size scaling says every pair crosses at one eta_c. Pairs that
    disagree are evidence AGAINST a critical point, and one averaged number
    hides exactly that."""
    rows = _synthetic(eta_c=10.0, sizes=(100, 200)) + [
        dict(r, n_agents=400, u4=r["u4"] - 0.25 * (r["eta"] - 13.0) / 10.0)
        for r in _synthetic(eta_c=13.0, sizes=(100,))]
    out = binder_crossing(rows)
    assert out["eta_c_spread"] is None or out["eta_c_spread"] >= 0.0
    assert len(out["pairs"]) == 2


def test_a_peak_on_the_grid_edge_is_flagged():
    """A sweep that did not bracket the transition reports its own edge as the
    location and looks exactly like a measurement (guardrail 4)."""
    rows = _synthetic(eta_c=10.0, etas=np.arange(3.0, 8.0, 1.0))
    peaks = susceptibility_peaks(rows)
    assert all(p["on_grid_edge"] for p in peaks)
    rep = audit_ladder([], rows)
    assert rep["peak_on_grid_edge"]
    assert rep["PASS"] is False


def test_the_audit_requires_the_ladder_to_bracket_both_phases():
    ordered_only = [r for r in _synthetic() if r["psi"] > 0.9]
    rep = audit_ladder([], ordered_only)
    assert rep["brackets_transition"] is False
    assert rep["PASS"] is False


def test_the_audit_flags_an_unequilibrated_cell():
    rows = _synthetic()
    rows[3]["max_abs_drift"] = DRIFT_TOLERANCE * 4
    rep = audit_ladder([], rows)
    assert rep["cells_not_equilibrated"]
    assert rep["PASS"] is False


def test_the_audit_refuses_a_ladder_that_mixed_substrates():
    """A criticality claim on two substrates is two claims (DOCS 5)."""
    rows = _synthetic()
    cells = [{"engine": "LeviathanEngine", "suite_version": "0.3.0"},
             {"engine": "ReferenceEngine", "suite_version": "0.3.0"}]
    rep = audit_ladder(cells, rows)
    assert rep["engines"] == ["LeviathanEngine", "ReferenceEngine"]
    assert rep["PASS"] is False


def test_a_flat_susceptibility_peak_is_reported_as_not_growing():
    rows = [dict(r, chi=1.0) for r in _synthetic()]
    rep = audit_ladder([{"engine": "E", "suite_version": "v"}], rows)
    assert rep["chi_peak_grows_with_n"] is False


# --------------------------------------------------------------------------
# the cell itself
# --------------------------------------------------------------------------

def test_a_cell_reports_both_susceptibilities_separately():
    """They differ by an order of magnitude and peak in the same place, which is
    what makes reporting one under the other's name an easy and invisible
    mistake."""
    rows = _synthetic()
    for r in rows:
        assert r["chi"] != r["chi_between_seed"] or r["chi"] == 0.0
    red = reduce_ladder([
        {"n_agents": 100, "eta": 1.0, "seed": s, "psi_mean": 0.5 + 0.01 * s,
         "chi": 2.0, "u4": 0.5, "drift": 0.0} for s in range(4)])
    assert len(red) == 1
    assert red[0]["chi"] == 2.0
    assert red[0]["chi_between_seed"] != red[0]["chi"]


def test_a_burn_in_that_leaves_no_samples_is_refused():
    """A variance over fewer than two samples is 0.0, which reads as a perfectly
    ordered swarm rather than as an empty measurement."""
    spec = CriticalitySpec(steps=10, burn_in=10, seeds=1)
    with pytest.raises(ValueError, match="burn_in"):
        run_cell(40, 5.0, 0, spec, engine=ReferenceEngine())


def test_a_cell_is_reproducible_and_names_its_substrate():
    spec = CriticalitySpec(steps=40, burn_in=20, seeds=1)
    a = run_cell(40, 6.0, 3, spec, engine=ReferenceEngine())
    b = run_cell(40, 6.0, 3, spec, engine=ReferenceEngine())
    assert a == b
    assert a["engine"] == "ReferenceEngine"
    assert a["n_samples"] == spec.steps - spec.burn_in


def test_the_noise_axis_actually_destroys_order():
    """The non-degeneracy floor. If psi does not fall with eta on this substrate
    there is no transition to validate and every column above is a tie."""
    spec = CriticalitySpec(steps=300, burn_in=150, seeds=1)
    quiet = run_cell(100, 0.0, 0, spec, engine=ReferenceEngine())
    loud = run_cell(100, 20.0, 0, spec, engine=ReferenceEngine())
    assert quiet["psi_mean"] > 0.9
    assert loud["psi_mean"] < 0.5


# --------------------------------------------------------------------------
# the delayed-view baseline wrapper
# --------------------------------------------------------------------------

def test_the_delay_axis_is_refused_if_the_config_key_was_not_understood():
    """This was a live no-op and it is why the assert exists.

    `DelayedView` reads `coupling_delay_steps`; writing the engine's spelling
    `comm_latency_steps` instead is silently ignored and yields delay 0. The
    whole ladder then comes back byte-identical at every delay, which reads as
    "delay does not affect ordering" -- the same unknown-key failure that left
    the `vicsek` algorithm block inert for months.
    """
    import numpy as np
    from gossamer.benchmarks.baselines import delayed_view_baseline, vicsek_alignment

    rng = np.random.default_rng(0)
    # The cap is the other way the request can be silently reduced.
    monkey = pytest.MonkeyPatch()
    monkey.setenv("DCC_MAX_DELAY_STEPS", "4")
    try:
        with pytest.raises(ValueError, match="resolved"):
            delayed_view_baseline(vicsek_alignment(), 40, 0.1, 8, rng)
    finally:
        monkey.undo()


def test_a_negative_delay_is_refused():
    import numpy as np
    from gossamer.benchmarks.baselines import delayed_view_baseline, vicsek_alignment
    with pytest.raises(ValueError):
        delayed_view_baseline(vicsek_alignment(), -1, 0.1, 8,
                              np.random.default_rng(0))


def test_zero_delay_is_exactly_the_undelayed_baseline():
    """The axis must reduce to the identity at its origin, or every delayed
    number is offset by a wrapper artefact rather than by delay."""
    import numpy as np
    from gossamer.benchmarks.baselines import delayed_view_baseline, vicsek_alignment

    pos = np.random.default_rng(3).normal(size=(20, 3))
    vel = np.random.default_rng(4).normal(size=(20, 3))
    plain = vicsek_alignment()(pos, vel, np.random.default_rng(0))
    wrapped = delayed_view_baseline(vicsek_alignment(), 0, 0.1, 20,
                                    np.random.default_rng(0))(
        pos, vel, np.random.default_rng(0))
    assert np.allclose(plain, wrapped, atol=1e-9)


def test_the_delay_axis_actually_moves_the_order_parameter():
    """Non-degeneracy. A ladder whose order parameter is pinned has measured
    nothing, and a pinned ladder is exactly what the key bug produced."""
    import numpy as np
    from gossamer.benchmarks.baselines import delayed_view_baseline, vicsek_alignment
    from gossamer.benchmarks.criticality import _psi

    def run(delay):
        scenario = VicsekTransitionScenario(num_agents=100)
        series = []
        base = delayed_view_baseline(vicsek_alignment(), delay, 0.1, 100,
                                     np.random.default_rng(1))
        run_benchmark(scenario, base, BenchmarkConfig(steps=300, seed=1),
                      engine=ReferenceEngine(),
                      observer=lambda s, p, v: series.append(_psi(v)) if s >= 150 else None)
        return float(np.mean(series))

    assert run(40) < run(0) - 0.01


def test_actuation_references_the_true_velocity_not_the_delayed_one():
    """Without the re-actuation the residual injects energy every step and the
    swarm diverges on a substrate that does not clamp speed -- which is all
    three of ours. The check is that speed stays bounded over a long delayed run.
    """
    import numpy as np
    from gossamer.benchmarks.baselines import delayed_view_baseline, vicsek_alignment

    scenario = VicsekTransitionScenario(num_agents=60)
    speeds = []
    base = delayed_view_baseline(vicsek_alignment(), 30, 0.1, 60,
                                 np.random.default_rng(2))
    run_benchmark(scenario, base, BenchmarkConfig(steps=400, seed=2),
                  engine=ReferenceEngine(),
                  observer=lambda s, p, v: speeds.append(
                      float(np.max(np.linalg.norm(v, axis=1)))))
    assert np.isfinite(speeds).all()
    # vicsek_alignment targets |v| = 1; anything an order of magnitude above it
    # is the energy-injection signature rather than a swarm moving fast.
    assert max(speeds) < 10.0


# --------------------------------------------------------------------------
# a crossing must be SIGNIFICANT, not merely a sign change
# --------------------------------------------------------------------------

def test_a_sign_change_where_the_curves_are_flat_is_not_a_crossing():
    """This was WRONG FIRST and the wrong version produced a confident number.

    In the ordered phase U4 is pinned at 2/3 to five decimals for every size, so
    the difference between two curves wanders at the 1e-5 level. Taking the first
    sign change from the left then reported eta_c = 0.0018 for a transition the
    susceptibility puts near eta = 12 -- a crossing manufactured out of scanning
    order.
    """
    from gossamer.benchmarks.criticality import SIGNIFICANT_U4
    rows = []
    for i, eta in enumerate(np.arange(0.0, 8.0, 1.0)):
        # Both sizes flat at 2/3, differing by a wobble far below the floor.
        wobble = (SIGNIFICANT_U4 / 100.0) * (1 if i % 2 else -1)
        rows.append({"n_agents": 100, "eta": float(eta), "n_seeds": 20,
                     "psi": 0.99, "psi_sd": 0.0, "chi": 0.01,
                     "chi_between_seed": 0.03, "u4": 2 / 3,
                     "u4_sd": 0.0, "max_abs_drift": 0.0})
        rows.append({"n_agents": 200, "eta": float(eta), "n_seeds": 20,
                     "psi": 0.99, "psi_sd": 0.0, "chi": 0.01,
                     "chi_between_seed": 0.03, "u4": 2 / 3 + wobble,
                     "u4_sd": 0.0, "max_abs_drift": 0.0})
    out = binder_crossing(rows)
    assert out["n_pairs_with_crossing"] == 0
    assert out["eta_c_mean"] is None


def test_a_sign_change_smaller_than_the_seed_noise_is_not_a_crossing():
    """The other half of the significance test: a difference that clears the
    absolute floor but not the measured seed spread is not resolved either."""
    rows = []
    for i, eta in enumerate(np.arange(0.0, 6.0, 1.0)):
        d = 0.02 * (1 if i % 2 else -1)
        for n, off in ((100, 0.0), (200, d)):
            rows.append({"n_agents": n, "eta": float(eta), "n_seeds": 20,
                         "psi": 0.5, "psi_sd": 0.0, "chi": 1.0,
                         "chi_between_seed": 3.0, "u4": 0.55 + off,
                         # sd large enough that 3 sigma exceeds the 0.02 swing
                         "u4_sd": 0.15, "max_abs_drift": 0.0})
    out = binder_crossing(rows)
    assert out["n_pairs_with_crossing"] == 0


def test_monotone_curves_are_reported_as_the_evidence_against_a_crossing():
    """When U4 is ordered by N at every eta the curves cannot have crossed inside
    the sweep, and that is the finding -- this is exactly what the measured
    ladder shows on the Vicsek substrate."""
    rows = []
    for eta in np.arange(8.0, 14.0, 1.0):
        for k, n in enumerate((100, 200, 400)):
            rows.append({"n_agents": n, "eta": float(eta), "n_seeds": 20,
                         "psi": 0.4, "psi_sd": 0.01, "chi": 1.0,
                         "chi_between_seed": 3.0,
                         "u4": 0.60 + 0.02 * k,      # strictly increasing in N
                         "u4_sd": 0.002, "max_abs_drift": 0.0})
    out = binder_crossing(rows)
    assert out["n_pairs_with_crossing"] == 0
    assert out["n_pairs_monotone"] == len(out["pairs"]) == 2


def test_a_crossing_on_a_grid_point_is_reported_once():
    """A crossing landing exactly ON a sampled eta has a zero difference there,
    and the intervals either side each see a sign change -- so one crossing
    arrives as two and the pair reads as ambiguous rather than resolved."""
    out = binder_crossing(_synthetic(eta_c=10.0, sizes=(100, 200)))
    assert out["pairs"][0]["n_significant_crossings"] == 1
    assert out["n_pairs_ambiguous"] == 0
    assert out["eta_c_mean"] == pytest.approx(10.0, abs=0.15)
