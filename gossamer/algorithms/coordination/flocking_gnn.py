"""
Boids flocking as a zero-parameter message-passing policy.

This is an architectural twin of the classical ``flock_step`` function in
:mod:`gossamer.algorithms.coordination.flocking`: identical math, but
expressed against the :class:`~gossamer.graph.MessagePassingPolicy`
interface so it can be dropped into the same training / evaluation loop
as a learned MAPPO policy. Use the functional version when you only
need one step of Boids; use this class when you're comparing classical
and learned policies on a common substrate.

The three classical forces (alignment, cohesion, separation) map to a
single per-edge message that carries all three contributions; the update
step applies the per-agent sum and clips to ``max_speed``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gossamer.graph import InteractionGraph, MessagePassingPolicy, build_radius_graph


@dataclass
class FlockingGNN(MessagePassingPolicy):
    """Hand-crafted Boids as a GNN policy.

    Set ``reduce`` to ``"sum"``: alignment and cohesion want a neighbor
    average (handled inside :meth:`message` by dividing by degree at
    aggregation time), but separation is a sum-of-inverse-squares. We
    aggregate raw vector contributions and let :meth:`update` do the
    per-node normalization — simpler than pre-dividing per edge.
    """

    alignment_weight: float = 1.0
    cohesion_weight: float = 1.0
    separation_weight: float = 1.5
    separation_distance: float = 1.0
    max_speed: float = 5.0
    dt: float = 0.1

    reduce: str = "sum"

    def message(self, graph: InteractionGraph) -> np.ndarray:
        """Emit per-edge contribution to [align, cohesion, separation, degree].

        Stacked as a single (E, 3*D + 1) message; the last column counts
        neighbors so :meth:`update` can mean-normalize alignment and cohesion.
        """
        if graph.num_edges == 0 or graph.velocities is None:
            d = graph.positions.shape[1]
            return np.zeros((0, 3 * d + 1), dtype=float)
        src = graph.edges[:, 0]
        dst = graph.edges[:, 1]
        pos = graph.positions
        vel = graph.velocities
        d = pos.shape[1]

        rel_pos = pos[dst] - pos[src]
        dist2 = np.einsum("ij,ij->i", rel_pos, rel_pos) + 1e-9

        # Alignment contribution: neighbor velocity (mean-normalized in update)
        align_c = vel[dst]

        # Cohesion contribution: neighbor position
        cohesion_c = pos[dst]

        # Separation contribution: -rel_pos / d^2 for neighbors inside separation_distance
        inside = dist2 < (self.separation_distance ** 2)
        sep_c = np.zeros_like(rel_pos)
        if np.any(inside):
            sep_c[inside] = -rel_pos[inside] / dist2[inside, None]

        ones = np.ones((align_c.shape[0], 1), dtype=float)
        return np.concatenate([align_c, cohesion_c, sep_c, ones], axis=1)

    def update(self, aggregated: np.ndarray, graph: InteractionGraph) -> np.ndarray:
        """Combine aggregated messages with the node's own state to produce
        a new velocity (per-agent acceleration is ``new_v - old_v`` / dt).
        """
        if graph.velocities is None:
            return np.zeros_like(graph.positions)
        d = graph.positions.shape[1]
        if aggregated.size == 0:
            return graph.velocities.copy()

        align_sum = aggregated[:, :d]
        cohesion_sum = aggregated[:, d:2 * d]
        sep_sum = aggregated[:, 2 * d:3 * d]
        counts = aggregated[:, -1]
        has_nbr = counts > 0

        avg_vel = np.zeros_like(align_sum)
        avg_vel[has_nbr] = align_sum[has_nbr] / counts[has_nbr, None]
        center = np.zeros_like(cohesion_sum)
        center[has_nbr] = cohesion_sum[has_nbr] / counts[has_nbr, None]

        align = self.alignment_weight * (avg_vel - graph.velocities)
        align[~has_nbr] = 0.0
        cohesion = self.cohesion_weight * (center - graph.positions)
        cohesion[~has_nbr] = 0.0
        separation = self.separation_weight * sep_sum

        new_vel = graph.velocities + align + cohesion + separation
        speed = np.linalg.norm(new_vel, axis=1)
        scale = np.where(speed > self.max_speed, self.max_speed / np.maximum(speed, 1e-12), 1.0)
        return new_vel * scale[:, None]


def flock_step_gnn(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float = 0.1,
    alignment_weight: float = 1.0,
    cohesion_weight: float = 1.0,
    separation_weight: float = 1.5,
    neighbor_radius: float = 10.0,
    separation_distance: float = 1.0,
    max_speed: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop-in replacement for :func:`flocking.flock_step` that routes through
    the GNN abstraction. Useful as a sanity baseline in benchmarks.

    Returns updated ``(positions, velocities)``.
    """
    graph = build_radius_graph(positions, neighbor_radius, velocities=velocities)
    policy = FlockingGNN(
        alignment_weight=alignment_weight,
        cohesion_weight=cohesion_weight,
        separation_weight=separation_weight,
        separation_distance=separation_distance,
        max_speed=max_speed,
        dt=dt,
    )
    new_velocities = policy.step(graph)
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


__all__ = ["FlockingGNN", "flock_step_gnn"]
