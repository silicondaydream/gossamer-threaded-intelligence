Gossamer Threaded Intelligence
==============================

Overview
Gossamer is the Arboria Labs policy library for large-scale, decentralized, multi-agent systems. It computes agent actions per step and exposes a consistent interface for coordination, navigation, resilience, and optimization. Gossamer is designed to be used with Leviathan Engine (physics, state evolution) and Maneuver.Map (orchestration, sweeps, visualization). On its own it can run in notebooks or scripts, but its full value and performance are realized when it is coupled to the rest of the Arboria toolchain.

Motivation
Massive swarms require local-first decision making that scales. Traditional centralized control fails under latency, bandwidth, and compute constraints. Gossamer exists to encode scalable, composable policies that trade global optimality for robustness, efficiency, and feasibility at extreme scale.

Novelty and usefulness
- Policy-first design: policies are pure step functions that can be evaluated inside a simulator, in a server, or on embedded compute.
- Scale-aware algorithms: emphasis on local interactions, bounded neighborhoods, and O(1) per-agent updates.
- Composability: policies can be combined (e.g., DMB + TF-ACO, ICCD + flocking) and instrumented with metrics.
- Research-to-production pipeline: the same codepaths that validate hypotheses in papers are used in deployed runs.

Important caveat
Gossamer is intended to be used with Leviathan Engine and Maneuver.Map. Without Leviathan, you lose physics, latency/field models, and high-throughput state stepping. Without Maneuver.Map, you lose orchestration, sweeps, and visualization. This repo is public-facing, but it is part of a toolkit and is not a standalone system by design.

Pre-requisites
- Python >= 3.8
- NumPy >= 1.24
- Pandas 2.x
- (Optional) Google Artifact Registry auth if installing the wheel
- (For full system) Leviathan Engine service and Maneuver.Map backend running

What’s new
- Stable distribution name: `gossamer-threaded-intelligence` with import `gossamer`
- Cloud Build pipelines for build, smoke test, and publish
- Spatial hashing option for large-N flocking

Install
- From Artifact Registry (recommended):
  - pip install keyrings.google-artifactregistry-auth
  - pip install --extra-index-url https://us-central1-python.pkg.dev/arboria-research/python-packages/simple/ gossamer-threaded-intelligence==0.1.0
- From source (dev):
  - pip install -r requirements.txt
  - pip install -e .

Core APIs and algorithms
- Coordination
  - Flocking (Boids baseline)
  - Consensus
  - Task allocation / auction allocation
- Navigation
  - Potential fields
- Communication
  - Broadcast, gossip protocols
- Optimization
  - PSO and related parameter search utilities
- Resilience
  - Self-healing topology primitives
- Metrics
  - cohesion, alignment, separation, and other swarm-quality measures

How it works (high level)
- Gossamer policies operate on the current agent state (positions, velocities, local observations).
- The policy outputs accelerations or action vectors.
- Leviathan applies physics, energy, and environment fields to advance the state.
- Maneuver.Map orchestrates repeated runs, parameter sweeps, and visualization.

Example: per-step action computation
```python
import numpy as np
from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.utils.metrics import cohesion

dt = 0.1
pos = np.random.randn(100, 3)
vel = np.zeros_like(pos)

for _ in range(100):
    _, vel = flock_step(
        pos, vel, dt,
        alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5,
        neighbor_radius=10.0, separation_distance=1.0, max_speed=5.0,
        use_spatial=True,
    )
    pos = pos + vel * dt
print('cohesion:', cohesion(pos))
```

Example: basic task allocation (with inline comments)
```python
import numpy as np
from gossamer.algorithms.coordination.task_allocation import auction_allocate_tasks

# Simulated 2D positions for haulers and depots
haulers = np.array([[0.0, 0.0], [10.0, 5.0], [4.0, -3.0]])
depots = np.array([[2.0, 1.0], [8.0, 6.0]])

# Assign each hauler to a depot index (min-cost matching heuristic)
assignments = auction_allocate_tasks(haulers, depots)

# Print which depot each hauler should target next
for i, depot_idx in enumerate(assignments):
    if depot_idx is None:
        print(f"hauler {i}: idle (no assignment)")
    else:
        print(f"hauler {i}: -> depot {int(depot_idx)} at {depots[int(depot_idx)]}")
```

How we've used Gossamer @ Arboria
- ICCD: CRDT-based intent propagation with DTN contact plans
- DMB / TF-ACO: emergent behavior tuning and coverage policies
- HMA: hierarchical market auctions for macro–micro coordination
- Flocking baselines for ablation and benchmarking

Research applications and discoveries
- Large-scale coordination under delayed/partitioned networks (ICCD)
- Emergent behavior phase diagrams (DMB + TF-ACO)
- Macro–micro synergy for construction/ISRU scenarios (HMA)
- Measurable improvements in coherence, coverage, energy efficiency, and resilience at scale

Compatibility
- Python >= 3.8; NumPy >= 1.24; Pandas 2.x
- Intended to be invoked from FastAPI backends (e.g., Maneuver.Map) or batch scripts

Roadmap
- GPU-accelerated kernels for large swarms
- Unified algorithm parameter schemas and validation
- Benchmark suite with reproducible baselines
- Additional coordination primitives and resilience strategies

Support
- Open an issue with a minimal repro (inputs, parameters, expected vs actual)
