"""
The Arboria Swarm Benchmark.

A fixed set of canonical scenarios with standard baselines, so that
every paper and every tool release reports against the same ground
truth. This is the moat that separates "we tuned flocking" from "we
made measurable progress against prior art."

Scenarios (each under :mod:`gossamer.benchmarks.scenarios`):

* ``dispersal`` — given a dense initial cluster, how fast can the swarm
  spread to a target density without colliding?
* ``rendezvous`` — given random initial positions, how fast can the
  swarm meet at a common point?
* ``coverage`` — explore a bounded region; maximize cells visited per
  unit time.
* ``leader_follower`` — one special agent is exogenously driven; does
  the rest of the swarm stay within a given distance?
* ``predator_prey`` — adversarial agents chase; measure survival and
  evasion.
* ``byzantine`` — inject k% silently faulty agents; measure robustness.

Baselines are defined in :mod:`gossamer.benchmarks.baselines` and must
be reported for every scenario:

* ``random`` — uniform random accelerations (lower bound).
* ``greedy`` — hand-crafted greedy solution per scenario.
* ``gossamer`` — the most appropriate classical policy for the
  scenario (flocking for rendezvous, TF-ACO for coverage, etc.).
* ``mappo`` — learned policy from :mod:`gossamer.learning.mappo`.

Use :func:`run_benchmark` / :func:`leaderboard` to execute and compare;
run :func:`generate_leaderboard_md` to emit the paper-ready table.
"""
from __future__ import annotations

from gossamer.benchmarks.harness import (
    BenchmarkConfig,
    BenchmarkResult,
    generate_leaderboard_md,
    leaderboard,
    run_benchmark,
)

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "generate_leaderboard_md",
    "leaderboard",
    "run_benchmark",
]
