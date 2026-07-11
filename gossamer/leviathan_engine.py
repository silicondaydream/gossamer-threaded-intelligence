"""Drive the benchmark on the real Leviathan engine, not just the reference.

`gossamer.benchmarks.harness` has always advertised that you can inject any
`PhysicsEngine` — "pass Leviathan ... to run the suite on the same substrate as the
papers". You could not. **No adapter existed.** Every benchmark number in the repo
came from `ReferenceEngine`, and a `ReferenceEngine` number is not comparable to a
paper (CLAUDE.md §1.3: "never compare a benchmark result to a paper unless both ran
on the same substrate"). That is a hole in the N2 moat: the benchmark's whole claim
is to be the neutral standard, and it could not run on the engine the standard is
about.

The near-miss that made it look solved: `gossamer.interfaces.leviathan_interface`
wraps an env with `reset()` / `step(actions)` / `compute_metrics()` — a completely
different shape, with no sim ids and an action *dict*. It cannot be handed to
`run_benchmark`.

Requires the compiled `leviathan` module (the `leviathan-base` image, or a local
cmake build). Import raises a clear error if it is absent rather than silently
falling back to the reference — a benchmark that quietly ran on a different
substrate than you asked for is the exact failure this file exists to close.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

__all__ = ["LeviathanEngine"]


class LeviathanEngine:
    """`PhysicsEngine` over the compiled C++ engine.

    Satisfies the 4-method protocol in `gossamer.engine`, so it drops straight into
    `run_benchmark(..., engine=LeviathanEngine())` and `leaderboard(engine=...)`.

    Note the substrate differences that make this NOT interchangeable with
    `ReferenceEngine` for a published number, even though both are "Leviathan
    kinematics": the C++ owns its own RNG (noise, faults), and its config is now
    validated — a key the reference would ignore raises `ConfigError` here. That
    strictness is the point; it is why a benchmark run on this engine is the one
    that can be compared to a paper.
    """

    def __init__(self) -> None:
        try:
            import leviathan  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "LeviathanEngine needs the compiled `leviathan` module, which is not "
                "importable here. Build it (cmake + pybind11) or run inside the "
                "`leviathan-base` image. Refusing to fall back to ReferenceEngine: a "
                "benchmark that silently ran on a different substrate than you asked "
                "for is not a benchmark."
            ) from exc
        self._leviathan = leviathan
        self._sims: Dict[str, object] = {}
        self._next = 0

    # -- PhysicsEngine ------------------------------------------------------

    def create_sim(self, config_map: Dict[str, str]) -> str:
        cfg = {str(k): str(v) for k, v in config_map.items()}
        # The harness leaves the channel unconfigured (scenarios score coordination,
        # not comms), and the engine's comm model is a no-op in that case. Faults
        # must be off explicitly: the engine's fault_prob default is 0.01, which
        # flips ~99.996% of a swarm faulty over a long run, and a benchmark that
        # silently killed its agents would just look like a bad policy.
        cfg.setdefault("fault_prob", "0.0")
        cfg.setdefault("energy_rate", "0.0")
        sim = self._leviathan.Simulation(cfg)
        sim_id = f"lev-{self._next}"
        self._next += 1
        self._sims[sim_id] = sim
        return sim_id

    def set_state(self, sim_id: str, pos: np.ndarray, vel: np.ndarray) -> None:
        sim = self._sims[sim_id]
        n = int(np.asarray(pos).shape[0])
        sim.set_state(
            np.asarray(pos, dtype=float).tolist(),
            np.asarray(vel, dtype=float).tolist(),
            [1.0] * n,     # energies: full, and energy_rate is 0 so they stay full
            [0] * n,       # faulty flags
            0,             # step index
        )

    def step(self, sim_id: str, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        sim = self._sims[sim_id]
        a = np.asarray(accel, dtype=float)
        # The C++ takes actions as {agent_index: [ax, ay, az]}.
        sim.step({i: a[i].tolist() for i in range(a.shape[0])})
        return (np.asarray(sim.get_positions(), dtype=float),
                np.asarray(sim.get_velocities(), dtype=float))

    def destroy(self, sim_id: str) -> None:
        self._sims.pop(sim_id, None)

    # -- extra (not in the Protocol, mirrors ReferenceEngine.metrics) --------

    def metrics(self, sim_id: str) -> Dict[str, float]:
        return dict(self._sims[sim_id].metrics())
