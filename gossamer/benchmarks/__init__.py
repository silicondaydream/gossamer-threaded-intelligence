"""
The Arboria Swarm Benchmark.

A fixed set of canonical scenarios with standard baselines, so that every paper and
every tool release reports against the same ground truth. This is the moat that
separates "we tuned flocking" from "we made measurable progress against prior art."

Scenarios (:mod:`gossamer.benchmarks.scenarios`, registry ``ALL_SCENARIOS``):

* ``dispersal`` — given a dense initial cluster, how fast can the swarm spread to a
  target density without colliding?
* ``rendezvous`` — given random initial positions, how fast can the swarm meet at a
  common point?
* ``coverage`` — explore a bounded region; maximize cells visited per unit time.
* ``leader_follower`` — one special agent is exogenously driven; does the rest of
  the swarm stay within a given distance?
* ``byzantine`` — a fraction of agents emit garbage intents (the harness replaces
  their commands via :meth:`Scenario.corrupt_actions`); measure robustness.

Baselines (:mod:`gossamer.benchmarks.baselines`, registry ``DEFAULT_BASELINES``):

* ``random`` — uniform random accelerations (lower bound).
* ``greedy`` — hand-crafted greedy solution per scenario.
* ``gossamer_flocking`` — the classical Boids policy from the shared kernels.

Use :func:`run_benchmark` / :func:`leaderboard` to execute and compare, and
:func:`generate_leaderboard_md` to emit the paper-ready table. Both take an
``engine=``: the default is :class:`gossamer.engine.ReferenceEngine` (pure NumPy),
and :class:`gossamer.leviathan_engine.LeviathanEngine` runs the suite on the real
C++ substrate — which is the only way a benchmark number is comparable to a paper.

Three things this docstring used to claim that were not true, all fixed rather than
reworded: it documented a ``predator_prey`` scenario that does not exist, it listed
a ``mappo`` baseline that was never wired into ``DEFAULT_BASELINES``, and it
described ``byzantine`` as injecting faulty agents when the harness never corrupted
anyone — the scenario marked its adversaries and nothing read the marks.
"""
from __future__ import annotations

from gossamer.benchmarks.baselines import DEFAULT_BASELINES, Baseline
from gossamer.benchmarks.harness import (
    BenchmarkConfig,
    BenchmarkResult,
    generate_leaderboard_md,
    leaderboard,
    run_benchmark,
)
from gossamer.benchmarks.scenarios import ALL_SCENARIOS, Scenario, ScenarioContext

__all__ = [
    "ALL_SCENARIOS",
    "DEFAULT_BASELINES",
    "Baseline",
    "BenchmarkConfig",
    "BenchmarkResult",
    "Scenario",
    "ScenarioContext",
    "generate_leaderboard_md",
    "leaderboard",
    "run_benchmark",
]
