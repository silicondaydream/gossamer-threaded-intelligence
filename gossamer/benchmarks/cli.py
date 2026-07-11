"""`gossamer-bench` — run the Arboria Swarm Benchmark from a shell.

The benchmark is the N2 moat, and a moat nobody can run is not a moat. This makes
the leaderboard reproducible by anyone with the wheel:

    gossamer-bench                                  # every scenario x baseline
    gossamer-bench --scenarios rendezvous byzantine --seeds 3
    gossamer-bench --engine leviathan               # paper-comparable substrate
    gossamer-bench --out leaderboard.md

`--engine leviathan` needs the compiled engine. It RAISES if that is unavailable
rather than falling back to the reference: a leaderboard that silently ran on a
different substrate than the one you asked for is not comparable to a paper, and
would be worse than no number at all (CLAUDE.md §1.3).
"""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from gossamer.benchmarks.baselines import DEFAULT_BASELINES
from gossamer.benchmarks.harness import (
    BenchmarkConfig,
    generate_leaderboard_md,
    leaderboard,
)
from gossamer.benchmarks.scenarios import ALL_SCENARIOS


def _build_engine(name: str):
    if name == "reference":
        return None  # the harness default
    if name == "leviathan":
        from gossamer.leviathan_engine import LeviathanEngine
        return LeviathanEngine()
    raise SystemExit(f"unknown engine {name!r} (reference|leviathan)")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="gossamer-bench", description=__doc__.splitlines()[0])
    p.add_argument("--scenarios", nargs="*", choices=sorted(ALL_SCENARIOS),
                   help="default: all")
    p.add_argument("--baselines", nargs="*", choices=sorted(DEFAULT_BASELINES),
                   help="default: all")
    p.add_argument("--engine", default="reference", choices=("reference", "leviathan"),
                   help="'leviathan' is the paper-comparable substrate")
    p.add_argument("--agents", type=int, default=500)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seeds", type=int, default=1, help="number of seeds per cell")
    p.add_argument("--out", help="write the Markdown leaderboard here (default: stdout)")
    args = p.parse_args(argv)

    cfg = BenchmarkConfig(num_agents=args.agents, steps=args.steps)
    scenarios = args.scenarios or sorted(ALL_SCENARIOS)
    results = leaderboard(
        scenarios=scenarios,
        baselines=args.baselines or sorted(DEFAULT_BASELINES),
        configs={s: cfg for s in scenarios},
        num_seeds=args.seeds,
        engine=_build_engine(args.engine),
    )

    md = generate_leaderboard_md(results)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(md)
        print(f"wrote {args.out} ({len(results)} cells, engine={args.engine})",
              file=sys.stderr)
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
