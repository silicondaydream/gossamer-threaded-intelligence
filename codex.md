
## Gossamer Codex

Gossamer is the algorithm library used by Leviathan and Maneuver.Map to drive agent behavior and assess performance.

Integration
- Use `gossamer.interfaces.leviathan_interface.LeviathanInterface` to run Gossamer logic against Leviathan’s Python env.
- Algorithms (e.g., `algorithms/coordination/flocking.py`) produce desired velocities; the orchestrator converts to accelerations per step.
- Metrics (cohesion, alignment, separation) are sampled during runs for fitness and analysis.

Local development
```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

Run with Leviathan
```bash
cd examples
PYTHONPATH=../../leviathan-engine python run_with_leviathan.py --steps 200 \
  --config ../../leviathan-engine/examples/simple_flock/minimal.cfg
```

Roadmap
- Expand algorithm coverage (consensus under delays, resilient task allocation, navigation with fields)
- Numba/JIT fast paths and optional GPU prototypes
- Richer metrics and experiment descriptors for publication‑grade analysis
- Documentation site with API and end‑to‑end examples
