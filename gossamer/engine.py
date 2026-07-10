"""The physics-substrate seam.

Gossamer must not own a simulation loop. It owns *algorithms*; Leviathan owns
physics. But the benchmark harness needs to step something, and depending on the
compiled C++ extension would make the benchmark suite unrunnable for anyone who
just `pip install`s the wheel — which defeats the purpose of a public benchmark.

So this module defines the minimal engine interface (:class:`PhysicsEngine`) plus
:class:`ReferenceEngine`, a pure-NumPy implementation whose semantics are pinned
to Leviathan's, term for term. A benchmark result is only comparable to a paper
result if the substrate agrees, and until now three substrates disagreed:

============  ==================  ===================  ==================
substrate     boundary            speed clamp          integrator
============  ==================  ===================  ==================
Leviathan     teleport to face    none                 verlet / euler
benchmarks    teleport to face    clamped (!)          euler only
FakeEngine    modulo wrap (!)     clamped (!)          euler only
============  ==================  ===================  ==================

``ReferenceEngine`` is the Leviathan column. The differences matter: a modulo wrap
preserves an agent's overshoot past the face while a teleport discards it (up to
``max_speed·dt`` of displacement per crossing), and a speed clamp silently
stabilises a policy that would diverge on the real engine.

Leviathan reference points, so the equivalence can be re-checked when the C++ moves:

* boundary — ``src/core/environment/environment.cpp`` (``if (p.x > bound_) p.x = -bound_;``)
* integrators — ``src/core/modules/physics_module.cpp`` (``apply_euler``, ``apply_velocity_verlet``)
* no speed clamp: ``acc_min``/``acc_max`` bound the *random* accel drawn for
  action-less agents, they do not clamp a commanded acceleration.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

__all__ = ["PhysicsEngine", "ReferenceEngine"]


@runtime_checkable
class PhysicsEngine(Protocol):
    """What a stepper must provide to drive a benchmark or an experiment."""

    def create_sim(self, config_map: Dict[str, str]) -> str: ...

    def set_state(self, sim_id: str, pos: np.ndarray, vel: np.ndarray) -> None: ...

    def step(self, sim_id: str, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...

    def destroy(self, sim_id: str) -> None: ...


def _wrap_periodic(pos: np.ndarray, bound: float) -> np.ndarray:
    """Leviathan's boundary rule: a coordinate past a face is placed *on* the
    opposite face, discarding the overshoot.

    This is not a modulo wrap. It is what the C++ does, so it is what the
    reference does. (A modulo wrap would be the more principled periodic boundary;
    changing the C++ would move the published P1/P3 numbers, so it is out of scope
    here and recorded as a known divergence from textbook periodic BCs.)
    """
    out = pos.copy()
    np.copyto(out, -bound, where=(pos > bound))
    np.copyto(out, bound, where=(pos < -bound))
    return out


class ReferenceEngine:
    """Pure-NumPy engine matching Leviathan's kinematics.

    Deliberately does **not** clamp speed: neither does the C++. A policy that
    diverges here diverges on the real engine, which is the point.
    """

    def __init__(self) -> None:
        self._sims: Dict[str, dict] = {}

    def create_sim(self, config_map: Dict[str, str]) -> str:
        cfg = {str(k): str(v) for k, v in (config_map or {}).items()}
        n = int(float(cfg.get("num_agents", 10)))
        dt = float(cfg.get("dt", 1.0))
        bound = float(cfg.get("bound", 100.0))
        integrator = cfg.get("integrator", "euler").lower()
        if integrator not in ("euler", "velocity_verlet"):
            raise ValueError(
                f"ReferenceEngine supports 'euler' and 'velocity_verlet', not {integrator!r}")
        seed = int(float(cfg.get("seed", 0)))
        rng = np.random.default_rng(seed)
        spread = float(cfg.get("init_spread", bound))
        sid = uuid.uuid4().hex
        self._sims[sid] = {
            "pos": rng.uniform(-spread, spread, size=(n, 3)),
            "vel": np.zeros((n, 3)),
            "dt": dt, "bound": bound, "integrator": integrator, "steps": 0,
        }
        return sid

    def set_state(self, sim_id: str, pos: np.ndarray, vel: np.ndarray) -> None:
        s = self._sims[sim_id]
        s["pos"] = np.asarray(pos, dtype=float).copy()
        s["vel"] = np.asarray(vel, dtype=float).copy()

    def step(self, sim_id: str, accel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = self._sims[sim_id]
        pos, vel, dt = s["pos"], s["vel"], s["dt"]
        a = np.zeros_like(pos) if accel is None else np.asarray(accel, dtype=float)

        if s["integrator"] == "velocity_verlet":
            pos = pos + vel * dt + 0.5 * a * dt * dt
            vel = vel + a * dt
        else:  # semi-implicit (symplectic) Euler: velocity first, then position
            vel = vel + a * dt
            pos = pos + vel * dt

        s["pos"] = _wrap_periodic(pos, s["bound"])
        s["vel"] = vel
        s["steps"] += 1
        return s["pos"], s["vel"]

    def metrics(self, sim_id: str) -> Dict[str, float]:
        return {"steps": float(self._sims[sim_id]["steps"])}

    def destroy(self, sim_id: str) -> None:
        self._sims.pop(sim_id, None)
