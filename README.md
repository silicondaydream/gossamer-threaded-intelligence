# Gossamer Threaded Intelligence

> Gossamer Threaded Intelligence is a suite of proprietary algorithms for swarm intelligence, distributed decision-making, and robust autonomous agent coordination.

## Installation

```bash
git clone <repo-url>
cd gossamer-intelligence
pip install -r requirements.txt
pip install -e .
```

## Structure

- gossamer/: main package
-   communication/: communication protocols (BaseProtocol, BroadcastProtocol, GossipProtocol)
- examples/: demonstration scripts (run_flocking_demo.py, run_task_allocation_demo.py)
- tests/: unit tests

## Usage Example

```python
from gossamer.simulator import SwarmSimulator

def log(step, positions, velocities, metrics):
    print(
        f"Step {step}: cohesion={metrics['cohesion']:.3f}, "
        f"alignment={metrics['alignment']:.3f}, "
        f"separation={metrics['separation']:.3f}"
    )

sim = SwarmSimulator(n_agents=30, n_dims=2, dt=0.1)
sim.run(50, callback=log)
```

For full documentation, see the `docs/` directory:

```bash
cd docs
pip install -r requirements.txt
sphinx-build -b html source build/html
```