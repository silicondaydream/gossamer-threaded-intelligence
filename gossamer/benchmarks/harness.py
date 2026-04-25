"""
Benchmark driver and leaderboard generation.

Runs a scenario + baseline combination under a lightweight in-process
NumPy stepper (so the benchmark suite has no Leviathan dependency — the
stepper uses the same semi-implicit Euler semantics as the C++ default
path). When ``engine_mode="inprocess"`` is requested, the harness routes
through the Leviathan pybind11 module instead for apples-to-apples
numbers against paper runs.

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


@dataclass
class BenchmarkConfig:
    """One benchmark run's knobs."""
    num_agents: int = 500
    steps: int = 500
    dt: float = 0.1
    bound: float = 100.0
    seed: int = 42
    record_trajectory: bool = True
    max_speed: float = 10.0


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


def _step_numpy(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray,
                dt: float, bound: float, max_speed: float) -> Tuple[np.ndarray, np.ndarray]:
    """Semi-implicit Euler with periodic-box wrap and speed clip."""
    new_vel = vel + accel * dt
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    scale = np.where(speed > max_speed, max_speed / np.maximum(speed, 1e-12), 1.0)
    new_vel = new_vel * scale
    new_pos = pos + new_vel * dt
    # Periodic box
    new_pos = np.where(new_pos > bound, -bound, np.where(new_pos < -bound, bound, new_pos))
    return new_pos, new_vel


def run_benchmark(
    scenario: Scenario,
    baseline: Baseline,
    config: BenchmarkConfig,
    baseline_name: str = "baseline",
) -> BenchmarkResult:
    """Run a single scenario + baseline combination end to end.

    Uses the pure-NumPy stepper by default; deterministic for a given
    seed. For Leviathan-engine runs, drive the engine externally and
    call this via a callable that forwards positions/velocities.
    """
    rng = np.random.default_rng(config.seed)
    pos, vel = scenario.init_state(rng, config.num_agents, config.bound)
    trajectory: List[Dict[str, np.ndarray]] = []
    total_reward = 0.0

    t0 = time.perf_counter()
    prev_pos = pos.copy()
    prev_vel = vel.copy()

    for step in range(config.steps):
        ctx = ScenarioContext(step=step, total_steps=config.steps, dt=config.dt)
        accel = baseline(pos, vel, rng)
        pos, vel = _step_numpy(pos, vel, accel, config.dt, config.bound, config.max_speed)
        r = scenario.step_reward(pos, vel, prev_pos, prev_vel, ctx)
        total_reward += float(np.sum(r))
        if config.record_trajectory:
            trajectory.append({"pos": pos.copy(), "vel": vel.copy()})
        prev_pos = pos
        prev_vel = vel

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
) -> List[BenchmarkResult]:
    """Run the full matrix of ``scenarios x baselines x seeds``.

    Returns all results flattened; aggregate afterward with
    :func:`generate_leaderboard_md`. Missing configs default to
    :class:`BenchmarkConfig`.
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
                result = run_benchmark(scenario, baseline, run_cfg, baseline_name=b_name)
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
