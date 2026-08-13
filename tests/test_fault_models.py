"""Standardized fault models for the benchmark (roadmap 18) — the substrate-free half.

The faults themselves live in the C++ engine, and Gossamer's CI does not build it,
so the rows that MEASURE damage live in `maneuver-map/tests/test_fault_ladder.py`
(whose CI does build it). Putting them here behind an `importorskip` would be the
exact move conftest.py forbids: under `ARBORIA_REQUIRE_TESTS=1` a skip is a failure,
so the tests would either go red in CI or be silently disarmed. What is testable
without the engine — the registry's shape and every refusal path — is tested here.

Almost every test here is an ANTI-NO-OP test, because that is this feature's
characteristic failure and the stack has shipped it before: `ByzantineScenario`
marked adversaries that nothing read, so the "byzantine" leaderboard row was
plain rendezvous under a different label. A fault row that silently contained no
faults does not look broken — it looks like a robustness result.
"""
import numpy as np
import pytest

from gossamer.benchmarks import ALL_SCENARIOS, DEFAULT_BASELINES, BenchmarkConfig, run_benchmark
from gossamer.benchmarks.faults import FAULT_MODELS, FaultModel, NoFaultsFiredError
from gossamer.engine import ReferenceEngine


def _baseline(scenario):
    b = DEFAULT_BASELINES["greedy"](scenario)
    return b["rendezvous"] if isinstance(b, dict) else b


# --- the registry -------------------------------------------------------------

def test_every_fault_model_names_evidence_except_the_control():
    """A model with nothing to point at cannot prove it happened, so it cannot be
    added. `none` is the single exception: there, nothing happening is the point."""
    for name, fm in FAULT_MODELS.items():
        if name == "none":
            assert fm.evidence_metric is None
            assert not fm.requires_engine_faults
        else:
            assert fm.evidence_metric, f"{name} has no evidence metric"
            assert fm.requires_engine_faults, f"{name} must declare its substrate need"


def test_the_ladder_is_ordered_by_bit_significance_in_the_spec():
    """The registry's claim is about WHICH bits flip, so the bit ranges themselves
    must be ordered even before anything is measured. The measured consequence is
    locked in maneuver-map's test_fault_ladder.py."""
    uniform = FAULT_MODELS["seu"].config
    significant = FAULT_MODELS["seu_significant"].config
    unprotected = FAULT_MODELS["seu_unprotected"].config
    # Identical rate across the ladder: only bit significance varies.
    assert uniform["seu_rate"] == significant["seu_rate"] == unprotected["seu_rate"]
    assert int(significant["seu_bit_low"]) > int(uniform["seu_bit_low"])
    assert int(unprotected["seu_bit_low"]) > int(significant["seu_bit_high"])
    # The exponent starts at 52; `seu` and `seu_significant` must stay below it.
    assert int(uniform["seu_bit_high"]) <= 51
    assert int(significant["seu_bit_high"]) <= 51
    assert int(unprotected["seu_bit_low"]) >= 52


def test_an_unknown_fault_model_is_refused():
    scenario = ALL_SCENARIOS["rendezvous"]()
    cfg = BenchmarkConfig(num_agents=8, steps=5, fault_model="cosmic_rays")
    with pytest.raises(ValueError, match="unknown fault model"):
        run_benchmark(scenario, _baseline(scenario), cfg, engine=ReferenceEngine())


# --- the no-op guards ---------------------------------------------------------

def test_a_fault_that_never_fires_is_refused_not_reported():
    """THE test. A run whose fault never fired must not reach the leaderboard,
    because a fault-free run under a fault label reads as robustness."""
    fm = FaultModel(name="never", doc="", config={}, evidence_metric="seu_flips_total")
    with pytest.raises(NoFaultsFiredError, match="fired ZERO faults"):
        fm.verify({"seu_flips_total": 0.0})
    assert fm.verify({"seu_flips_total": 3.0}) == 3.0


def test_a_metric_the_engine_cannot_report_is_refused():
    """An absent metric is indistinguishable from a fault that did not happen — so
    it is refused rather than read as zero."""
    with pytest.raises(NoFaultsFiredError, match="does not expose"):
        FAULT_MODELS["seu"].verify({"num_agents": 10.0})


def test_a_fault_row_refuses_a_substrate_without_a_fault_module():
    """ReferenceEngine has no fault module, so a fault row against it would apply
    nothing and report a clean fault-free run. That is the silent no-op, and it is
    why the measuring tests live where a real engine is guaranteed."""
    scenario = ALL_SCENARIOS["rendezvous"]()
    cfg = BenchmarkConfig(num_agents=8, steps=5, fault_model="seu")
    with pytest.raises(NoFaultsFiredError):
        run_benchmark(scenario, _baseline(scenario), cfg, engine=ReferenceEngine())


def test_the_control_runs_on_any_substrate_and_reports_no_faults():
    scenario = ALL_SCENARIOS["rendezvous"]()
    cfg = BenchmarkConfig(num_agents=8, steps=5, fault_model="none")
    r = run_benchmark(scenario, _baseline(scenario), cfg, engine=ReferenceEngine())
    assert r.fault_model == "none"
    assert r.faults_fired == 0.0
