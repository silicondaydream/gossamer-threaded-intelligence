Gossamer Threaded Intelligence
==============================

## Overview

Gossamer is Arboria Labs' policy and analysis library for large-scale,
decentralized, multi-agent systems. It exposes:

- **Classical coordination primitives** — flocking (Boids), consensus,
  auction-based task allocation, potential fields, gossip, broadcast,
  voting mechanisms — all pure NumPy, all reproducible.
- **A graph-based message-passing substrate** (`gossamer.graph`) so
  hand-crafted policies and learned GNN policies share one interface.
- **An information-theoretic, criticality, and spectral metric suite**
  (`gossamer.metrics`) for paper-grade phase-transition analysis.
- **A PyTorch MARL toolkit** (`gossamer.learning`) with a parameter-
  shared CTDE actor-critic base, a learnable comm channel with
  bandwidth / latency / loss / energy accounting, a reference MAPPO
  driver, and a domain-randomization harness.
- **The Arboria Swarm Benchmark** (`gossamer.benchmarks`) — canonical
  scenarios and baselines every new policy must report against.

Gossamer is designed to pair with the [Leviathan
Engine](https://www.arborialabs.com/tools/leviathan_engine) (physics
core) and [Maneuver.Map](https://www.arborialabs.com/tools/maneuver_map)
(orchestration + visualization). It also runs standalone — the benchmark
harness and the classical algorithms have no physics dependency.

## What's new in 0.2.0

- **`gossamer.graph`** — `InteractionGraph`, `MessagePassingPolicy`,
  `build_radius_graph`. The substrate used by both classical and learned
  policies.
- **`gossamer.algorithms.coordination.flocking_gnn`** and
  **`consensus_gnn`** — classical algorithms expressed as zero-parameter
  GNN layers. Drop-in comparators for learned policies.
- **`gossamer.metrics.info`** — transfer entropy, mutual information
  (histogram + KSG estimator).
- **`gossamer.metrics.criticality`** — susceptibility, Binder cumulant,
  correlation length, branching ratio, avalanche statistics.
- **`gossamer.metrics.graph`** — algebraic connectivity, spectral gap,
  clustering coefficient, structural summary.
- **`gossamer.learning`** — PyTorch MARL toolkit (import requires
  `torch`; the rest of Gossamer stays NumPy-only).
  - `CommChannel` + `CommChannelConfig` — learnable channel with
    bandwidth, latency, Bernoulli loss, energy accounting.
  - `GraphActorCritic` + `SharedMLPHead` — parameter-shared CTDE policy.
  - `mappo` — reference MAPPO driver + GAE.
  - `domain_randomization` — PettingZoo env wrapper that resamples
    physical parameters per episode.
- **`gossamer.benchmarks`** — canonical scenarios (dispersal,
  rendezvous, coverage, leader-follower, byzantine) plus a standard
  baseline set (`random`, `greedy`, `gossamer_flocking`). Driver +
  Markdown leaderboard generator.
- **`gossamer.utils.spatial`** — shared uniform-grid neighbor query;
  used by the classical flocking path and the Maneuver.Map runner so
  there's one implementation, not two.
- **RNG discipline** — every stochastic algorithm now accepts an
  explicit `rng: np.random.Generator | int | None` argument. Legacy
  module-level `np.random.*` calls and `import random` usages are
  gone; `GossipProtocol`'s old `random_state` kwarg still works as a
  back-compat alias.

## Motivation

Massive swarms require local-first decision making that scales.
Traditional centralized control fails under latency, bandwidth, and
compute constraints. Gossamer encodes scalable, composable policies
that trade global optimality for robustness, efficiency, and
feasibility at extreme scale.

## Novelty and usefulness

- **Policy-first design**: policies are pure step functions or GNN
  layers that can be evaluated inside a simulator, in a server, or on
  embedded compute.
- **Scale-aware algorithms**: emphasis on local interactions, bounded
  neighborhoods, and O(1) per-agent updates.
- **Unified substrate**: the same `InteractionGraph` / `MessagePassingPolicy`
  interface supports hand-crafted Boids (as a zero-parameter GNN layer)
  and a learned MAPPO policy — ablations are apples-to-apples.
- **Measurement-first**: information-theoretic and criticality metrics
  are first-class, not afterthoughts.
- **Research-to-production pipeline**: the same codepaths that validate
  hypotheses in papers are used in deployed runs.

## Pre-requisites

- Python >= 3.8
- NumPy >= 1.24
- Pandas 2.x
- scipy (for KSG mutual information and NSGA-III ranking)
- networkx (for self-healing topology)
- scikit-learn (optional)
- torch (optional — required only for `gossamer.learning`)
- gymnasium + pettingzoo (optional — required for env wrappers that
  consume learned policies; lives in the Leviathan repo, not here)
- Google Artifact Registry auth if installing the wheel

## Install

- From Artifact Registry (recommended):
  - `pip install keyrings.google-artifactregistry-auth`
  - `pip install --extra-index-url https://us-central1-python.pkg.dev/arboria-research/python-packages/simple/ gossamer-threaded-intelligence==0.2.0`
- From source (dev):
  - `pip install -r requirements.txt`
  - `pip install -e .`

## Core APIs and algorithms

- **Coordination**
  - Flocking (Boids baseline) — classical and as a GNN layer
  - Consensus (Laplacian average) — classical and as a GNN layer
  - Task allocation — Hungarian + ε-auction
- **Navigation**
  - Potential fields
- **Communication**
  - Broadcast, push/pull gossip
- **Decision-making**
  - Voting (plurality, Borda, approval, Condorcet, Schulze)
- **Optimization**
  - PSO (seeded via `rng`)
- **Resilience**
  - Self-healing topology primitives
- **Graph + Learning + Benchmarks**
  - `gossamer.graph` — interaction-graph abstractions
  - `gossamer.metrics` — info-theoretic, criticality, spectral
  - `gossamer.learning` — MARL toolkit (PyTorch)
  - `gossamer.benchmarks` — scenarios + baselines + leaderboard

## Example 1: per-step action computation (classical)

```python
import numpy as np
from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.utils.metrics import cohesion

dt = 0.1
rng = np.random.default_rng(0)
pos = rng.normal(scale=10, size=(100, 3))
vel = np.zeros_like(pos)

for _ in range(100):
    pos, vel = flock_step(
        pos, vel, dt,
        alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5,
        neighbor_radius=10.0, separation_distance=1.0, max_speed=5.0,
        use_spatial=True,
    )
print('cohesion:', cohesion(pos))
```

## Example 2: same step via the GNN interface

```python
from gossamer.graph import build_radius_graph
from gossamer.algorithms.coordination.flocking_gnn import FlockingGNN

policy = FlockingGNN(alignment_weight=1.0, cohesion_weight=1.0,
                    separation_weight=1.5, max_speed=5.0, dt=0.1)
graph = build_radius_graph(pos, radius=10.0, velocities=vel)
new_vel = policy.step(graph)
```

## Example 3: benchmark a policy

```python
from gossamer.benchmarks import leaderboard, generate_leaderboard_md
results = leaderboard(num_seeds=3)  # default scenarios + baselines
print(generate_leaderboard_md(results))
```

## Example 4: auction task allocation

```python
import numpy as np
from gossamer.algorithms.coordination.task_allocation import auction_allocate_tasks

haulers = np.array([[0.0, 0.0], [10.0, 5.0], [4.0, -3.0]])
depots  = np.array([[2.0, 1.0], [8.0, 6.0]])
assignments = auction_allocate_tasks(haulers, depots)
for i, depot_idx in enumerate(assignments):
    print(f"hauler {i}: depot {depot_idx}")
```

## Reproducibility

Every stochastic API accepts an `rng: np.random.Generator | int | None`
argument. Seeding a single `np.random.default_rng(seed)` and threading
it through produces byte-identical outputs on a given environment. See
the [reproducibility runbooks](https://www.arborialabs.com/research/reproducibility)
for the per-paper conventions.

## How we've used Gossamer @ Arboria

- **ICCD**: CRDT-based intent propagation with DTN contact plans
  ([paper](https://www.arborialabs.com/research/distributed_intelligence_interstellar_systems)).
- **DMB / TF-ACO**: density-modulated Boids and stigmergic coverage
  with phase-transition instrumentation
  ([paper](https://www.arborialabs.com/research/emergent_behavior_models_large_scale_space_exploration)).
- **HMA**: hierarchical market auctions for macro–micro coordination
  ([paper](https://www.arborialabs.com/research/macro_micro_synergy_planetary_engineering)).

## Compatibility

- Python >= 3.8; NumPy >= 1.24; Pandas 2.x
- Intended to be invoked from FastAPI backends (e.g., Maneuver.Map) or
  batch scripts.
- `gossamer.learning` additionally requires PyTorch 2.x.
- `gossamer.benchmarks` is pure NumPy; no learning deps.

## Roadmap

- Differentiable policies via JAX (Brax backend; see `leviathan_diff_brax` — **experimental, unvalidated stub**, not yet usable for results).
- Graph world models (JEPA / Dreamer / graph-WM) layered on top of
  `gossamer.learning`.
- Byzantine-robust MARL baselines in the benchmark suite.
- Formal-methods hooks (STL robustness) for differentiable swarm shaping.

## Support

- Open an issue with a minimal repro (inputs, parameters, expected vs
  actual).
