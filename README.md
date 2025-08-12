# Gossamer Threaded Intelligence

Lightweight Python library of swarm intelligence algorithms and utilities that drive agent behavior inside the Leviathan Engine and visualize via Maneuver.Map. Use it standalone for algorithm prototyping, or plug it into Leviathan for high‑performance, multi‑agent simulations at scale.

## How It Fits With Leviathan + Maneuver.Map

- Gossamer: Implements agent decision logic (e.g., flocking, task allocation) and metrics.
- Leviathan: High‑performance C++ simulator that executes actions from Gossamer and logs state.
- Maneuver.Map: Orchestrates experiments, tunes parameters, stores data, and renders 3D visualizations.

## Features

- Coordination algorithms: flocking, task allocation, consensus, navigation, resilience.
- Metrics: cohesion, alignment, separation, and helpers for evaluation.
- Interfaces: adapter to run Gossamer logic against Leviathan (`LeviathanInterface`).
- Examples: quick demos and integration scripts.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

Standalone prototype:
```python
from gossamer.simulator import SwarmSimulator

sim = SwarmSimulator(n_agents=30, dt=0.1)
sim.run(50, callback=lambda s,p,v,m: print(s, m))
```

Run against Leviathan (Python bindings):
```bash
cd examples
PYTHONPATH=../../leviathan-engine python run_with_leviathan.py --steps 200 \
  --config ../../leviathan-engine/examples/simple_flock/minimal.cfg
```

## API Highlights

- `gossamer.algorithms.coordination.flocking.flock_step(positions, velocities, dt, ...) -> (new_pos, new_vel)`
- `gossamer.utils.metrics.{cohesion, alignment, separation}`
- `gossamer.interfaces.leviathan_interface.LeviathanInterface(env)` implements a sim‑like loop over a Leviathan environment with `reset/step/compute_metrics` methods.

## Use Cases

- Research and teach swarm behaviors with reproducible metrics.
- Rapidly iterate on agent algorithms before scaling to millions of agents in Leviathan.
- Plug into Maneuver.Map’s evolutionary tuner to find optimal parameters across generations.

## Repo Structure

- `gossamer/algorithms/*` algorithm modules
- `gossamer/utils/metrics.py` evaluation metrics
- `gossamer/interfaces/leviathan_interface.py` Leviathan adapter
- `examples/` runnable demos
