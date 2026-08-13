"""The single-substrate story (roadmap 18): the benchmark runs on Leviathan.

The suite used to default to `ReferenceEngine` — pure NumPy, no compiled
dependency, runnable by anyone with the wheel. That convenience cost the
benchmark its central claim. `leviathan_engine.py` states the hole plainly:
*every benchmark number in the repo came from ReferenceEngine*, while DOCS §4
forbids comparing a benchmark result to a paper unless both ran on the same
substrate. A neutral standard that cannot run on the engine the standard is about
is not a standard.

So the default is now Leviathan and there is **no fallback**. That is the design:
a silent substrate swap leaves no trace in the number it produces, so it has to be
impossible rather than discouraged.

**Measured, and worth stating because it is counter-intuitive:** on the fault-free
kinematic scenarios the two substrates agree to within one ULP, and the re-baseline
therefore moved no number. `ReferenceEngine` is pinned to Leviathan's kinematics and
it turns out to be faithfully pinned. So the switch is NOT justified by the
arithmetic — it is justified by CAPABILITY: the reference has no fault module, no
SEU, and no config validation, so every row that needs those is a silent no-op on
it. That is what these tests pin, along with the default itself and the substrate
label that now travels with every result.
"""
import numpy as np
import pytest

from gossamer.benchmarks import ALL_SCENARIOS, DEFAULT_BASELINES, BenchmarkConfig, run_benchmark
from gossamer.benchmarks.harness import SUITE_VERSION, default_engine
from gossamer.engine import ReferenceEngine

leviathan = pytest.importorskip(
    "leviathan", reason="the benchmark's default substrate is the compiled engine")

from gossamer.leviathan_engine import LeviathanEngine  # noqa: E402


def _cell(engine=None, scenario="rendezvous", baseline="greedy", steps=120, seed=7):
    sc = ALL_SCENARIOS[scenario]()
    b = DEFAULT_BASELINES[baseline](sc)
    bl = b[scenario] if isinstance(b, dict) else b
    cfg = BenchmarkConfig(num_agents=40, steps=steps, dt=0.1, bound=100.0, seed=seed)
    return run_benchmark(sc, bl, cfg, baseline_name=baseline, engine=engine)


def test_the_default_substrate_is_the_compiled_engine():
    assert isinstance(default_engine(), LeviathanEngine)
    assert _cell().engine == "LeviathanEngine"


def test_every_result_names_the_substrate_that_produced_it():
    """A ReferenceEngine number and a Leviathan number are not comparable, and
    nothing else in the row says which one you are holding."""
    assert _cell(engine=ReferenceEngine()).engine == "ReferenceEngine"
    assert _cell(engine=LeviathanEngine()).engine == "LeviathanEngine"


def test_every_result_names_its_suite_version():
    """Results are comparable only within one (version, substrate) pair. The
    version moved when the substrate did, so pre-0.2.0 leaderboards are retired
    rather than merely stale."""
    assert _cell().suite_version == SUITE_VERSION == "0.2.0"


def test_the_substrates_agree_on_the_fault_free_kinematic_core():
    """MEASURED, and it is the opposite of what you might assume: on a noise-free,
    fault-free scenario the two substrates agree to within one ULP (3.7e-16
    relative on this cell; bit-identical on others).

    That is not a defect, it is `ReferenceEngine`'s contract — DOCS calls it "a
    pure-NumPy stepper pinned to Leviathan's kinematics" — and it is worth locking,
    because it is what makes the reference a legitimate development substrate at
    all. It also means the re-baseline moved no kinematic number.

    An earlier draft of this file asserted the substrates *disagreed*, and it
    passed — on that single ULP. That is a tie dressed as a difference, the exact
    thing guardrail 4 exists to catch, and it would have flipped on any compiler
    change. The real reason the default had to move is the next test."""
    ref = _cell(engine=ReferenceEngine())
    lev = _cell(engine=LeviathanEngine())
    assert ref.metric == pytest.approx(lev.metric, rel=1e-12), (
        f"the substrates have drifted apart: ref={ref.metric!r} lev={lev.metric!r}. "
        f"ReferenceEngine is supposed to be pinned to Leviathan's kinematics, so a "
        f"real gap here means one of them changed its physics.")


def test_the_substrate_matters_because_the_reference_cannot_express_faults():
    """THE load-bearing test, and the actual reason the default moved.

    The substrates agree on plain kinematics, so the switch buys nothing *there*.
    What it buys is everything the reference does not implement: `ReferenceEngine`
    has no fault module, no SEU, no validated config. A fault row against it would
    apply nothing and report a clean fault-free run — a silent no-op that reads as
    robustness. So the requirement is real exactly where the engines differ in
    CAPABILITY rather than in arithmetic."""
    sc = ALL_SCENARIOS["rendezvous"]()
    b = DEFAULT_BASELINES["greedy"](sc)
    bl = b["rendezvous"] if isinstance(b, dict) else b
    # Long enough for the upset rate to actually fire — a short cell would raise
    # NoFaultsFiredError on BOTH engines and the test would pass for the wrong reason.
    cfg = BenchmarkConfig(num_agents=60, steps=300, dt=0.1, bound=100.0,
                          fault_model="seu")
    from gossamer.benchmarks.faults import NoFaultsFiredError
    with pytest.raises(NoFaultsFiredError):
        run_benchmark(sc, bl, cfg, engine=ReferenceEngine())
    # ...and the same cell is fine on the substrate that implements it.
    r = run_benchmark(ALL_SCENARIOS["rendezvous"](), bl, cfg, engine=LeviathanEngine())
    assert r.faults_fired > 0


def test_reference_engine_is_still_reachable_for_development():
    """The compiled engine is the default, not a wall. Asking for the reference
    explicitly stays legal — it just cannot happen by accident, and the result
    records that it happened."""
    r = _cell(engine=ReferenceEngine())
    assert np.isfinite(r.metric)
    assert r.engine == "ReferenceEngine"


def test_a_leaderboard_cannot_mix_substrates_row_to_row():
    """One engine is constructed per leaderboard and shared across every cell, so
    a table cannot silently contain rows from two substrates — which would be
    invisible in the rendered Markdown."""
    from gossamer.benchmarks import leaderboard
    results = leaderboard(scenarios=["rendezvous"], baselines=["greedy", "random"],
                          configs={"rendezvous": BenchmarkConfig(
                              num_agents=20, steps=40, dt=0.1, bound=100.0)})
    assert results
    assert {r.engine for r in results} == {"LeviathanEngine"}
    assert {r.suite_version for r in results} == {SUITE_VERSION}
