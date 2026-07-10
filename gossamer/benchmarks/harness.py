"""
Benchmark driver and leaderboard generation.

Runs a scenario + baseline combination against an injected
:class:`gossamer.engine.PhysicsEngine`. The default is
:class:`~gossamer.engine.ReferenceEngine`, a pure-NumPy stepper whose kinematics
are pinned to Leviathan's, so the suite has no compiled dependency and can be run
by anyone who installs the wheel. Pass Leviathan itself (or Maneuver.Map's
``EngineClient``) to produce numbers directly comparable with paper runs.

This used to hardcode a private ``_step_numpy`` and its docstring claimed an
``engine_mode="inprocess"`` Leviathan path that **did not exist** — no such
parameter was ever defined. Worse, ``_step_numpy`` clamped speed while Leviathan
does not, so the benchmark silently stabilised policies that diverge on the real
engine. A benchmark whose substrate differs from the papers' cannot be the
neutral standard it exists to be.

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
from gossamer.engine import PhysicsEngine, ReferenceEngine


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
    extra: Dict[str, float] = field(default_factory=dict)


def run_benchmark(
    scenario: Scenario,
    baseline: Baseline,
    config: BenchmarkConfig,
    baseline_name: str = "baseline",
    engine: Optional[PhysicsEngine] = None,
) -> BenchmarkResult:
    """Run a single scenario + baseline combination end to end.

    ``engine`` defaults to :class:`~gossamer.engine.ReferenceEngine` (pure NumPy,
    Leviathan-equivalent kinematics) and is deterministic for a given seed. Pass
    Leviathan — or any object satisfying :class:`~gossamer.engine.PhysicsEngine` —
    to run the suite on the same substrate as the papers.

    The scenario owns the initial state, so the engine is created and then
    ``set_state``'d rather than being allowed to randomise its own.
    """
    engine = engine or ReferenceEngine()
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
    })
    engine.set_state(sim_id, pos, vel)

    t0 = time.perf_counter()
    prev_pos = pos.copy()
    prev_vel = vel.copy()

    try:
        for step in range(config.steps):
            ctx = ScenarioContext(step=step, total_steps=config.steps, dt=config.dt)
            accel = baseline(pos, vel, rng)
            pos, vel = engine.step(sim_id, accel)
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            r = scenario.step_reward(pos, vel, prev_pos, prev_vel, ctx)
            total_reward += float(np.sum(r))
            if config.record_trajectory:
                trajectory.append({"pos": pos.copy(), "vel": vel.copy()})
            prev_pos = pos
            prev_vel = vel
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
    :class:`BenchmarkConfig`. ``engine`` is forwarded to every cell, so a whole
    leaderboard can be regenerated on Leviathan for paper-comparable numbers.
    """
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
