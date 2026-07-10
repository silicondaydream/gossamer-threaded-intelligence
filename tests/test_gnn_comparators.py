"""The GNN comparators and the domain-randomization wrapper.

These modules are public API — named in the README, asserted by the publish gate
in `cloudbuild.publish.yaml` — but had **zero tests** and nothing in the repo
imports them. A cleanup audit nearly deleted them as dead code. They are not dead;
they are the "drop-in comparators for learned policies" the README advertises, and
the N2 benchmark's stated classical baseline. So test them, or the next audit will
be right to bin them.

`FlockingGNN` claims to be Boids "expressed as a zero-parameter GNN layer". That is
a checkable claim, and it holds to floating-point noise: its acceleration matches
`boids_accel_edges` to ~1e-14, the two differing only in reduction order (a
per-agent sum over directed edges vs `np.add.at` over `i<j` pairs).
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.consensus_gnn import AverageConsensusGNN
from gossamer.algorithms.coordination.flocking_gnn import FlockingGNN
from gossamer.algorithms.coordination.kernels import boids_accel_edges, kdtree_edges
from gossamer.graph import build_radius_graph
from gossamer.learning.domain_randomization import (
    DomainRandomizationConfig, LinearCurriculum,
)

BOIDS = dict(alignment_weight=1.0, cohesion_weight=1.0, separation_weight=1.5,
             separation_distance=1.0, max_speed=5.0)


def _state(n=24, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-8, 8, (n, 3)), rng.uniform(-1, 1, (n, 3))


# --------------------------------------------------------------------------
# FlockingGNN — a zero-parameter layer that must equal the classical kernel
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 3, 11])
@pytest.mark.parametrize("reduce", ["sum", "mean"])
def test_flocking_gnn_is_exactly_the_vectorised_boids_kernel(seed, reduce):
    pos, vel = _state(seed=seed)
    radius, dt = 5.0, 0.1

    graph = build_radius_graph(pos, radius, velocities=vel)
    new_vel = FlockingGNN(dt=dt, reduce=reduce, **BOIDS).step(graph)
    gnn_accel = (new_vel - vel) / dt

    eu, ev = kdtree_edges(pos, radius)
    kernel_accel = boids_accel_edges(
        pos, vel, eu, ev, dt, BOIDS["alignment_weight"], BOIDS["cohesion_weight"],
        BOIDS["separation_weight"], BOIDS["separation_distance"], BOIDS["max_speed"])

    # Not `array_equal`: the GNN sums per-agent over directed edges while the
    # kernel uses `np.add.at` over `i<j` pairs, so the reductions associate
    # differently. The tolerance is floating-point, not algorithmic (~1e-14).
    assert np.allclose(gnn_accel, kernel_accel, atol=1e-12)


def test_flocking_gnn_respects_the_speed_clamp():
    pos, vel = _state()
    graph = build_radius_graph(pos, 5.0, velocities=vel)
    new_vel = FlockingGNN(max_speed=2.0, dt=0.1).step(graph)
    assert np.all(np.linalg.norm(new_vel, axis=1) <= 2.0 + 1e-9)


def test_flocking_gnn_leaves_isolated_agents_coasting():
    pos = np.array([[0.0, 0, 0], [1000.0, 0, 0]])
    vel = np.array([[1.0, 0, 0], [0.0, 1.0, 0]])
    graph = build_radius_graph(pos, 1.0, velocities=vel)
    assert np.allclose(FlockingGNN(dt=0.1).step(graph), vel)


def test_flocking_gnn_without_velocities_is_a_zero_field():
    pos, _ = _state()
    graph = build_radius_graph(pos, 5.0)  # velocities omitted
    assert np.allclose(FlockingGNN(dt=0.1).step(graph), 0.0)


def test_radius_graph_is_directed_so_each_pair_appears_twice():
    """The GNN aggregates over directed edges; the kernel over `i<j` pairs. The
    two agree only because the kernel accumulates each pair into both endpoints."""
    pos, _ = _state()
    directed = build_radius_graph(pos, 5.0).num_edges
    undirected = kdtree_edges(pos, 5.0)[0].size
    assert directed == 2 * undirected


# --------------------------------------------------------------------------
# AverageConsensusGNN
# --------------------------------------------------------------------------

def _consensus_positions(pos, steps, radius=50.0):
    gnn = AverageConsensusGNN()
    out = [pos]
    for _ in range(steps):
        g = build_radius_graph(out[-1], radius)
        g.node_features = out[-1]
        out.append(gnn.step(g))
    return out


def test_consensus_gnn_monotonically_contracts_the_spread():
    rng = np.random.default_rng(1)
    traj = _consensus_positions(rng.uniform(-10, 10, (30, 3)), steps=6)
    spread = [float(p.var(axis=0).mean()) for p in traj]
    assert spread[-1] < spread[0]
    assert all(b <= a + 1e-9 for a, b in zip(spread, spread[1:]))


def test_consensus_gnn_preserves_the_centroid():
    """Average consensus draws agents together, it does not drag them anywhere."""
    rng = np.random.default_rng(2)
    pos = rng.uniform(-10, 10, (20, 3))
    after = _consensus_positions(pos, steps=1)[-1]
    assert np.allclose(after.mean(axis=0), pos.mean(axis=0), atol=1e-9)


def test_consensus_gnn_is_a_no_op_without_edges():
    pos = np.array([[0.0, 0, 0], [500.0, 0, 0]])
    g = build_radius_graph(pos, 1.0)
    g.node_features = pos
    assert np.array_equal(AverageConsensusGNN().step(g), pos)


# --------------------------------------------------------------------------
# Domain randomization
# --------------------------------------------------------------------------

def test_linear_curriculum_interpolates_easy_to_hard_and_clamps():
    c = LinearCurriculum(parameter="comm_loss_prob", easy=(0.0, 0.1),
                         hard=(0.4, 0.9), n_episodes=10)
    assert c.current(0) == pytest.approx((0.0, 0.1))
    assert c.current(5) == pytest.approx((0.2, 0.5))
    assert c.current(10) == pytest.approx((0.4, 0.9))
    assert c.current(10_000) == pytest.approx((0.4, 0.9)), "must clamp past the horizon"


def test_linear_curriculum_never_extrapolates_backwards():
    c = LinearCurriculum(parameter="fault_prob", easy=(0.0, 0.0),
                         hard=(0.1, 0.2), n_episodes=8)
    assert c.current(-5) == pytest.approx((0.0, 0.0))


def test_domain_randomization_config_defaults_to_randomising_nothing():
    cfg = DomainRandomizationConfig()
    ranges = (cfg.comm_latency_steps, cfg.comm_loss_prob, cfg.fault_prob,
              cfg.num_agents, cfg.field_strength)
    assert all(r is None for r in ranges), (
        "an unconfigured wrapper must be the identity, or a run silently picks up "
        "randomised physics it never asked for")
