"""
Benchmark driver and leaderboard generation.

Runs a scenario + baseline combination against an injected
:class:`gossamer.engine.PhysicsEngine`.

**The default substrate is Leviathan** (:class:`gossamer.leviathan_engine.LeviathanEngine`),
the compiled engine the papers run on. It used to be
:class:`~gossamer.engine.ReferenceEngine` — pure NumPy, no compiled dependency,
"runnable by anyone who installs the wheel" — and that convenience quietly cost
the benchmark its whole claim. `leviathan_engine.py` states the hole plainly:
*every benchmark number in the repo came from ReferenceEngine*, and DOCS is
explicit that you must never compare a benchmark result to a paper unless both ran
on the same substrate. A neutral standard that cannot run on the engine the
standard is about is not a standard.

So the convenient default is gone. If the compiled engine is missing, this RAISES
and names the opt-out rather than falling back — a silent substrate swap is the
precise failure being closed, and it is invisible in the result. Passing
``engine=ReferenceEngine()`` explicitly is still fine for development; the result
records which engine ran (``BenchmarkResult.engine``) so a reference number can
never be mistaken for a paper-comparable one.

Two substrate differences are why this is not cosmetic: the C++ owns its own RNG
(noise, faults), and its config is validated, so a key the reference silently
ignores raises here. A third one already bit: the original ``_step_numpy``
clamped speed while Leviathan does not, so the benchmark silently stabilised
policies that diverge on the real engine.

Output shape:

* :class:`BenchmarkResult` holds scenario metric, wall-clock, and any
  per-step traces.
* :func:`leaderboard` aggregates results across ``(scenario, baseline)``.
* :func:`generate_leaderboard_md` emits a paper-ready Markdown table.
"""
from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from gossamer.benchmarks.baselines import Baseline, DEFAULT_BASELINES
from gossamer.benchmarks.scenarios import ALL_SCENARIOS, Scenario, ScenarioContext
from gossamer.benchmarks.faults import FAULT_MODELS, FaultModel, NoFaultsFiredError
from gossamer.engine import PhysicsEngine, ReferenceEngine

#: Results are comparable only within one (version, substrate) pair. This is the
#: same discipline `orrery.benchmarks` follows — changing a frozen parameter is a
#: new version, never a knob.
#:
#: 0.2.0: the default substrate moved from ReferenceEngine to Leviathan, which
#: changes every kinematic number in the suite.
#: 0.3.0: two definition changes the 0.2.0 reference run found in the suite ITSELF.
#:   (1) `coverage` counts occupancy over the simulated VOLUME; it had indexed
#:       (y, x) only, and that projection is what left the row with no headroom —
#:       three of four baselines within 4% of the ceiling and the greedy walker
#:       answering no fault rung at all.
#:   (2) A scenario declares its own `required_config` instead of trusting the
#:       caller, so `vicsek_transition` can no longer be run at a density where
#:       nothing can order. Numbers do not move for (2) — the 0.2.0 generator
#:       already supplied that config externally; the change is that omitting it
#:       is now impossible rather than merely documented.
#: 0.2.0 leaderboards are retired by (1), not merely stale.
SUITE_VERSION = "0.3.0"


def default_engine() -> PhysicsEngine:
    """The compiled engine, or a refusal that names the opt-out.

    Never falls back to ReferenceEngine. A benchmark that silently ran on a
    different substrate than the papers is the exact failure this default exists
    to close, and it leaves no trace in the number it produces.
    """
    from gossamer.leviathan_engine import LeviathanEngine
    return LeviathanEngine()


@dataclass
class BenchmarkConfig:
    """One benchmark run's knobs."""
    num_agents: int = 500
    steps: int = 500
    dt: float = 0.1
    bound: float = 100.0
    seed: int = 42
    record_trajectory: bool = True
    # Advisory only. Leviathan does not clamp speed, so neither does the harness;
    # baselines that want a speed limit must enforce it in the acceleration they
    # return. The old stepper clamped here, which quietly rescued policies that
    # diverge on the real engine.
    max_speed: float = 10.0
    integrator: str = "euler"
    #: Name from `gossamer.benchmarks.faults.FAULT_MODELS`. "none" is the
    #: fault-free control. Any other value REQUIRES an engine with a fault module
    #: (LeviathanEngine) and refuses the run if the fault left no trace — see that
    #: module for why the evidence check is not optional.
    fault_model: str = "none"


@dataclass
class BenchmarkResult:
    scenario: str
    baseline: str
    metric: float
    mean_reward: float
    elapsed_sec: float
    num_agents: int
    steps: int
    seed: int
    #: The fault regime this row ran under, and the measured evidence it actually
    #: fired. Both travel with the result: a fault row whose count is absent from
    #: the record cannot be distinguished later from a fault-free one.
    fault_model: str = "none"
    faults_fired: float = 0.0
    #: Which substrate produced this number, and which suite version it belongs to.
    #: Both travel with the result because a ReferenceEngine number and a Leviathan
    #: number are not comparable, and nothing else in the row says which you have.
    engine: str = "unknown"
    suite_version: str = SUITE_VERSION
    extra: Dict[str, float] = field(default_factory=dict)


class ScenarioConfigConflictError(ValueError):
    """A caller set a config value the scenario declares it cannot run at."""


def apply_required_config(scenario: Scenario,
                          config: BenchmarkConfig) -> BenchmarkConfig:
    """Overlay ``scenario.required_config`` onto ``config``, refusing conflicts.

    A scenario that is only meaningful at a particular configuration says so on
    itself, and the harness applies it — so a caller cannot omit it. The failure
    this closes was silent and shipped: `leaderboard()` had always accepted
    per-scenario configs, nothing ever supplied one, and the substrate-validation
    row therefore ran at a density where no substrate can order. It returned a
    plausible number on a row whose baselines still differed, so the cross-baseline
    discrimination check passed it.

    Overriding a value the caller *deliberately* chose would be its own silent
    failure, so the two cases are separated: a key still at the
    :class:`BenchmarkConfig` default is filled in, and a key the caller set to
    something else is REFUSED. Requesting a run at a configuration the scenario
    declares invalid is a question the harness cannot answer honestly, and the
    answer it would otherwise give is a number under the wrong label.
    """
    required = getattr(scenario, "required_config", None)
    if not required:
        return config
    defaults = BenchmarkConfig()
    updates: Dict[str, object] = {}
    for key, want in required.items():
        if not hasattr(defaults, key):
            raise ScenarioConfigConflictError(
                f"scenario {scenario.name!r} requires unknown config key {key!r}")
        have = getattr(config, key)
        if have == want:
            continue
        if have != getattr(defaults, key):
            raise ScenarioConfigConflictError(
                f"scenario {scenario.name!r} requires {key}={want!r} but the caller "
                f"set {key}={have!r}. This scenario is not valid at that value — see "
                f"its docstring. Pass the default to accept the required value.")
        updates[key] = want
    return BenchmarkConfig(**{**config.__dict__, **updates}) if updates else config


def run_benchmark(
    scenario: Scenario,
    baseline: Baseline,
    config: BenchmarkConfig,
    baseline_name: str = "baseline",
    engine: Optional[PhysicsEngine] = None,
) -> BenchmarkResult:
    """Run a single scenario + baseline combination end to end.

    ``engine`` defaults to :func:`default_engine` — the COMPILED engine, which
    raises with build instructions rather than falling back. Deterministic for a
    given seed. Pass :class:`~gossamer.engine.ReferenceEngine` explicitly to use
    the pure-NumPy path; it cannot carry a fault row (no fault module), so the
    harness refuses that combination rather than reporting a fault-free run
    under a fault label.

    The scenario owns the initial state, so the engine is created and then
    ``set_state``'d rather than being allowed to randomise its own.
    """
    engine = engine if engine is not None else default_engine()
    # Applied HERE rather than in `leaderboard()` so it holds for every caller —
    # a scenario's validity requirement that only the leaderboard path honours is
    # the same trust-the-caller arrangement that failed the first time.
    config = apply_required_config(scenario, config)
    if config.fault_model not in FAULT_MODELS:
        raise ValueError(
            f"unknown fault model {config.fault_model!r}; expected one of "
            f"{sorted(FAULT_MODELS)}")
    fault: FaultModel = FAULT_MODELS[config.fault_model]
    # A fault model against a substrate with no fault module would apply nothing and
    # report a clean fault-free run under a fault label — the exact silent no-op the
    # registry exists to prevent. `metrics()` is how a fault proves it fired, so an
    # engine without it cannot carry a fault row.
    if fault.requires_engine_faults and not hasattr(engine, "metrics"):
        raise NoFaultsFiredError(
            f"fault model {config.fault_model!r} needs an engine that implements faults "
            f"and reports them; {type(engine).__name__} does not expose metrics(). Use "
            f"LeviathanEngine — ReferenceEngine has no fault module, so this row would "
            f"silently be a fault-free run.")

    rng = np.random.default_rng(config.seed)
    pos, vel = scenario.init_state(rng, config.num_agents, config.bound)
    trajectory: List[Dict[str, np.ndarray]] = []
    total_reward = 0.0

    sim_id = engine.create_sim({
        "num_agents": str(config.num_agents), "dt": str(config.dt),
        "bound": str(config.bound), "seed": str(config.seed),
        "integrator": config.integrator,
        # The channel is off: benchmark scenarios score coordination, and a run
        # with no comm keys makes the engine's comm model a no-op.
        **fault.config,
    })
    engine.set_state(sim_id, pos, vel)

    t0 = time.perf_counter()
    prev_pos = pos.copy()
    prev_vel = vel.copy()

    try:
        for step in range(config.steps):
            ctx = ScenarioContext(step=step, total_steps=config.steps, dt=config.dt)
            accel = baseline(pos, vel, rng)
            # The adversary acts on the COMMAND, not the state: a Byzantine agent
            # emits a garbage intent, it does not teleport. Identity for every
            # honest scenario. This call is what makes ByzantineScenario real —
            # without it the scenario marked its adversaries and nobody read the
            # marks, so the "byzantine" leaderboard row was plain rendezvous.
            accel = scenario.corrupt_actions(accel, rng, ctx)
            pos, vel = engine.step(sim_id, accel)
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            r = scenario.step_reward(pos, vel, prev_pos, prev_vel, ctx)
            total_reward += float(np.sum(r))
            if config.record_trajectory:
                trajectory.append({"pos": pos.copy(), "vel": vel.copy()})
            prev_pos = pos
            prev_vel = vel
        # Read the evidence BEFORE the sim is destroyed. `verify` raises when the
        # fault left no trace, so a fault row cannot reach the leaderboard without
        # having demonstrably contained faults.
        faults_fired = fault.verify(
            engine.metrics(sim_id) if hasattr(engine, "metrics") else {})
    finally:
        engine.destroy(sim_id)

    elapsed = time.perf_counter() - t0
    metric = scenario.terminal_metric(trajectory)

    return BenchmarkResult(
        scenario=scenario.name,
        baseline=baseline_name,
        metric=float(metric),
        mean_reward=float(total_reward / max(1, config.steps * config.num_agents)),
        elapsed_sec=float(elapsed),
        num_agents=config.num_agents,
        steps=config.steps,
        seed=config.seed,
        fault_model=config.fault_model,
        faults_fired=float(faults_fired),
        engine=type(engine).__name__,
    )


def leaderboard(
    scenarios: Optional[List[str]] = None,
    baselines: Optional[List[str]] = None,
    configs: Optional[Dict[str, BenchmarkConfig]] = None,
    num_seeds: int = 1,
    engine: Optional[PhysicsEngine] = None,
) -> List[BenchmarkResult]:
    """Run the full matrix of ``scenarios x baselines x seeds``.

    Returns all results flattened; aggregate afterward with
    :func:`generate_leaderboard_md`. Missing configs default to
    :class:`BenchmarkConfig`. ``engine`` is forwarded to every cell; ONE engine is
    built here and shared, so a leaderboard cannot silently mix substrates
    row-to-row. Defaults to Leviathan — see the module docstring for why the
    convenient pure-NumPy default was removed.
    """
    engine = engine if engine is not None else default_engine()
    scenarios = scenarios or list(ALL_SCENARIOS.keys())
    baselines = baselines or list(DEFAULT_BASELINES.keys())
    configs = configs or {}

    results: List[BenchmarkResult] = []
    for s_name in scenarios:
        scenario_cls = ALL_SCENARIOS[s_name]
        cfg = configs.get(s_name, BenchmarkConfig())
        for b_name in baselines:
            baseline_factory = DEFAULT_BASELINES[b_name]
            for seed_offset in range(num_seeds):
                run_cfg = BenchmarkConfig(**{**cfg.__dict__, "seed": cfg.seed + seed_offset})
                # Re-instantiate scenario so stateful scenarios reset between runs
                scenario = scenario_cls() if callable(scenario_cls) else scenario_cls
                baseline = baseline_factory(scenario)
                result = run_benchmark(scenario, baseline, run_cfg,
                                       baseline_name=b_name, engine=engine)
                results.append(result)
    return results


def _aggregate(results: List[BenchmarkResult]) -> Dict[Tuple[str, str], Dict[str, float]]:
    agg: Dict[Tuple[str, str], List[BenchmarkResult]] = {}
    for r in results:
        agg.setdefault((r.scenario, r.baseline), []).append(r)
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for key, rs in agg.items():
        metrics = [r.metric for r in rs]
        rewards = [r.mean_reward for r in rs]
        elapsed = [r.elapsed_sec for r in rs]
        out[key] = {
            "metric_mean": statistics.mean(metrics),
            "metric_std": statistics.pstdev(metrics) if len(metrics) > 1 else 0.0,
            "reward_mean": statistics.mean(rewards),
            "elapsed_mean": statistics.mean(elapsed),
            "seeds": float(len(rs)),
        }
    return out


def generate_leaderboard_md(results: List[BenchmarkResult]) -> str:
    """Emit a Markdown table ready to paste into a paper / page."""
    if not results:
        return "# Arboria Swarm Benchmark\n\n_No results._\n"
    agg = _aggregate(results)
    scenarios = sorted({k[0] for k in agg.keys()})
    baselines = sorted({k[1] for k in agg.keys()})

    lines: List[str] = []
    lines.append("# Arboria Swarm Benchmark — Leaderboard")
    lines.append("")
    lines.append("Terminal metric (mean ± std over seeds); see scenario docs for direction of better.")
    lines.append("")
    header = ["Scenario"] + [f"{b}" for b in baselines]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for s in scenarios:
        row = [s]
        for b in baselines:
            stats = agg.get((s, b))
            if stats is None:
                row.append("—")
            else:
                row.append(f"{stats['metric_mean']:.3f} ± {stats['metric_std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Wall-clock per run (seconds, mean):")
    lines.append("")
    lines.append("| " + " | ".join(["Scenario"] + baselines) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(baselines))) + "|")
    for s in scenarios:
        row = [s]
        for b in baselines:
            stats = agg.get((s, b))
            row.append(f"{stats['elapsed_mean']:.2f}" if stats else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "generate_leaderboard_md",
    "leaderboard",
    "run_benchmark",
]
