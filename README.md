Gossamer Threaded Intelligence
==============================

Overview
Gossamer is a Python library of decentralized, multi‑agent algorithms and utilities used to compute agent actions per step in large‑scale simulations. It integrates with the Leviathan Engine (physics) and the Maneuver.Map orchestrator (experiments + visualization).

What’s new
- Distributed as a versioned Python wheel in Artifact Registry (no source copying)
- Stable import name: `gossamer` (distribution: `gossamer-threaded-intelligence`)
- Cloud Build pipelines for build, smoke test, and publish

Install
- From Artifact Registry (recommended):
  - pip install keyrings.google-artifactregistry-auth
  - pip install --extra-index-url https://us-central1-python.pkg.dev/arboria-research/python-packages/simple/ gossamer-threaded-intelligence==0.1.0
- From source (dev):
  - pip install -r requirements.txt
  - pip install -e .

Core APIs
- Flocking
  - from gossamer.algorithms.coordination.flocking import flock_step
  - new_pos, new_vel = flock_step(pos, vel, dt, alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5, neighbor_radius=10.0, separation_distance=1.0, max_speed=5.0)
- Metrics
  - from gossamer.utils.metrics import cohesion, alignment, separation

Example (per‑step action computation)
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
    )
    pos = pos + vel * dt
print('cohesion:', cohesion(pos))
```

Publishing a new version
- Update version in setup.py
- Push changes and trigger `cloudbuild.publish.yaml`
  - Builds the wheel, smoke‑tests an import, uploads to the Artifact Registry Python repo

Compatibility
- Python >= 3.8; NumPy >= 1.24; Pandas 2.x
- Designed to be invoked from FastAPI backends (e.g., Maneuver.Map) or scripts

Roadmap
- Additional coordination primitives (rendezvous, coverage)
- GPU‑accelerated variants for large swarms
- Benchmark suite and reproducible evaluations

Support
- Open an issue with a minimal repro (inputs, parameters, expected vs actual)
