"""ReferenceEngine must match Leviathan term-for-term, or the benchmark cannot be
compared to a paper run.

Cross-references (re-check these when the C++ moves):
  * boundary   — src/core/environment/environment.cpp:137-142
  * euler      — src/core/modules/physics_module.cpp  apply_euler
  * verlet     — src/core/modules/physics_module.cpp  apply_velocity_verlet
"""
import numpy as np
import pytest

from gossamer.benchmarks.harness import BenchmarkConfig, run_benchmark
from gossamer.benchmarks.scenarios import ALL_SCENARIOS
from gossamer.engine import PhysicsEngine, ReferenceEngine


def _sim(engine, n=3, dt=1.0, bound=100.0, integrator="euler"):
    return engine.create_sim({"num_agents": str(n), "dt": str(dt),
                              "bound": str(bound), "integrator": integrator,
                              "seed": "1"})


def test_semi_implicit_euler_updates_velocity_before_position():
    """Leviathan's apply_euler: v += a*dt, THEN p += v*dt (so p sees the new v)."""
    e = ReferenceEngine()
    sid = _sim(e)
    e.set_state(sid, np.zeros((3, 3)), np.zeros((3, 3)))
    a = np.tile([1.0, 0.0, 0.0], (3, 1))
    pos, vel = e.step(sid, a)
    assert vel[0, 0] == pytest.approx(1.0)
    assert pos[0, 0] == pytest.approx(1.0)  # not 0.0 — an explicit Euler would give 0


def test_velocity_verlet_updates_position_before_velocity():
    e = ReferenceEngine()
    sid = _sim(e, integrator="velocity_verlet")
    e.set_state(sid, np.zeros((3, 3)), np.zeros((3, 3)))
    a = np.tile([1.0, 0.0, 0.0], (3, 1))
    pos, vel = e.step(sid, a)
    assert pos[0, 0] == pytest.approx(0.5)   # v*dt + 0.5*a*dt^2
    assert vel[0, 0] == pytest.approx(1.0)


def test_boundary_teleports_to_the_opposite_face_and_discards_overshoot():
    """Leviathan sets the coordinate to -bound exactly; it does NOT modulo-wrap.

    FakeEngine in Maneuver.Map does `((p + b) % 2b) - b`, which preserves the
    overshoot. The two substrates therefore disagree by up to one step of
    displacement per boundary crossing.
    """
    e = ReferenceEngine()
    sid = _sim(e, bound=10.0)
    e.set_state(sid, np.array([[9.0, 0.0, 0.0]]), np.array([[5.0, 0.0, 0.0]]))
    pos, _ = e.step(sid, np.zeros((1, 3)))  # would land at x=14 -> past bound
    assert pos[0, 0] == pytest.approx(-10.0)   # teleport, not -6.0 (modulo)


def test_speed_is_not_clamped():
    """Leviathan has no speed clamp; the old benchmark stepper did, which quietly
    stabilised policies that diverge on the real engine."""
    e = ReferenceEngine()
    sid = _sim(e, bound=1e12)
    e.set_state(sid, np.zeros((1, 3)), np.zeros((1, 3)))
    for _ in range(10):
        pos, vel = e.step(sid, np.array([[1000.0, 0.0, 0.0]]))
    assert vel[0, 0] == pytest.approx(10_000.0)


def test_reference_engine_satisfies_the_protocol():
    assert isinstance(ReferenceEngine(), PhysicsEngine)


def test_unknown_integrator_raises():
    with pytest.raises(ValueError, match="velocity_verlet"):
        _sim(ReferenceEngine(), integrator="rk4")


def test_benchmark_accepts_an_injected_engine():
    """The whole point of M6.7: the suite runs on whatever substrate you hand it."""
    calls = {"steps": 0}

    class SpyEngine(ReferenceEngine):
        def step(self, sim_id, accel):
            calls["steps"] += 1
            return super().step(sim_id, accel)

    scenario = ALL_SCENARIOS["rendezvous"]()
    from gossamer.benchmarks.baselines import DEFAULT_BASELINES
    baseline = DEFAULT_BASELINES["gossamer_flocking"](scenario)
    cfg = BenchmarkConfig(num_agents=10, steps=7, record_trajectory=True)

    result = run_benchmark(scenario, baseline, cfg, engine=SpyEngine())
    assert calls["steps"] == 7
    assert np.isfinite(result.metric)


def test_benchmark_is_deterministic_for_a_seed():
    scenario_cls = ALL_SCENARIOS["rendezvous"]
    from gossamer.benchmarks.baselines import DEFAULT_BASELINES
    cfg = BenchmarkConfig(num_agents=12, steps=15)

    def once():
        s = scenario_cls()
        return run_benchmark(s, DEFAULT_BASELINES["gossamer_flocking"](s), cfg).metric

    assert once() == pytest.approx(once())
