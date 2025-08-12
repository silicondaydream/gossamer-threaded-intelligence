#!/usr/bin/env python3
"""
Run Gossamer against Leviathan via the LeviathanEnv adapter.

Leviathan handles physics and logging; this script shows the integration layer
that matches Gossamer's simulator-style API for quick experiments.

Usage:
  PYTHONPATH=../leviathan-engine python run_with_leviathan.py --steps 100 --config ./leviathan.cfg
"""
import argparse
import os
from gossamer.interfaces.leviathan_interface import LeviathanInterface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100, help="Simulation steps")
    ap.add_argument("--config", type=str, default=None, help="Path to Leviathan config (key: value)")
    args = ap.parse_args()

    # Import LeviathanEnv from leviathan-engine repo (expect PYTHONPATH set accordingly)
    try:
        # Expect PYTHONPATH to include leviathan-engine/src/python
        from leviathan_env import LeviathanEnv
    except Exception as e:
        raise SystemExit(
            "Failed to import LeviathanEnv. Ensure PYTHONPATH includes leviathan-engine/src/python"
        ) from e

    cfg = None
    if args.config:
        # LeviathanEnv accepts a dict config; parse simple key: value file
        cfg = {}
        with open(args.config, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    cfg[k.strip()] = v.strip()

    env = LeviathanEnv(cfg)
    adapter = LeviathanInterface(env)

    def cb(step, pos, vel, metrics):
        if step % 10 == 0:
            print(f"Step {step}: metrics={metrics}")

    adapter.run(args.steps, callback=cb)
    print("Done.")


if __name__ == "__main__":
    main()
