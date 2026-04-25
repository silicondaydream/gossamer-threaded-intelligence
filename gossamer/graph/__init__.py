"""
Graph abstractions for message-passing swarm policies.

Swarm coordination algorithms — Boids, consensus, stigmergic ACO, ICCD
intent propagation — all share the same compute pattern: for each agent,
gather features from a local neighborhood and combine them into a new
state. This module introduces a single ``InteractionGraph`` object and a
``MessagePassingPolicy`` base class that every algorithm can plug into.

Why this matters:

* Zero-parameter policies (hand-crafted Boids, consensus) become GNN layers
  with the same shape as learned policies, so ablation against MARL
  baselines works with no adapter code.
* The edge set can be built once per step from a spatial grid and reused
  across coordination, communication, and metric computation — the
  dominant cost for large N.
* Heterogeneous agents become node-feature variants instead of a second
  class of code path.

The implementation is intentionally NumPy-only. A PyTorch mirror lives in
``gossamer.learning`` for differentiable policies; keeping the classical
interface dependency-free avoids forcing every user to install torch.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gossamer.utils.spatial import build_grid, neighbors_within


@dataclass
class InteractionGraph:
    """A snapshot of the swarm's interaction topology at one timestep.

    Parameters
    ----------
    positions:
        Node position array of shape ``(N, D)``.
    velocities:
        Node velocity array of shape ``(N, D)``. ``None`` when velocities
        are not modeled.
    edges:
        ``(E, 2)`` int array of directed edges ``(src, dst)``. Undirected
        graphs double-list each edge. Self-loops are optional.
    edge_features:
        Optional ``(E, F_e)`` array of per-edge features (relative
        position, distance, etc.). Built lazily by :func:`compute_edge_features`.
    node_features:
        Optional ``(N, F_n)`` array of per-node features beyond position /
        velocity — e.g. role, SOC, AoI, pheromone level.
    """

    positions: np.ndarray
    velocities: Optional[np.ndarray] = None
    edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    edge_features: Optional[np.ndarray] = None
    node_features: Optional[np.ndarray] = None

    @property
    def num_nodes(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edges.shape[0])

    def with_edge_features(self, include_distance: bool = True,
                           include_relative_position: bool = True,
                           include_relative_velocity: bool = False) -> "InteractionGraph":
        """Return a copy with ``edge_features`` populated from the node state."""
        return InteractionGraph(
            positions=self.positions,
            velocities=self.velocities,
            edges=self.edges,
            edge_features=compute_edge_features(
                self.positions,
                self.velocities,
                self.edges,
                include_distance=include_distance,
                include_relative_position=include_relative_position,
                include_relative_velocity=include_relative_velocity,
            ),
            node_features=self.node_features,
        )


def build_radius_graph(
    positions: np.ndarray,
    radius: float,
    velocities: Optional[np.ndarray] = None,
    include_self: bool = False,
) -> InteractionGraph:
    """Construct a radius-based interaction graph via the shared spatial grid.

    Every agent has an edge to each neighbor strictly within ``radius``.
    Edges are directed (i.e., ``(src, dst)`` with both directions listed)
    so downstream aggregation can remain asymmetric if the algorithm
    requires it. For O(1)-per-node cost, ``radius`` should be much less
    than the simulation extent.
    """
    if positions.size == 0:
        return InteractionGraph(positions=positions, velocities=velocities)
    grid, cell_idx = build_grid(positions, max(radius, 1e-9))
    src: list[int] = []
    dst: list[int] = []
    n = positions.shape[0]
    for i in range(n):
        nbrs = neighbors_within(positions, cell_idx, grid, radius, i)
        for j in nbrs:
            src.append(i)
            dst.append(j)
        if include_self:
            src.append(i)
            dst.append(i)
    edges = np.stack([np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)], axis=1) \
        if src else np.zeros((0, 2), dtype=np.int64)
    return InteractionGraph(positions=positions, velocities=velocities, edges=edges)


def compute_edge_features(
    positions: np.ndarray,
    velocities: Optional[np.ndarray],
    edges: np.ndarray,
    include_distance: bool = True,
    include_relative_position: bool = True,
    include_relative_velocity: bool = False,
) -> np.ndarray:
    """Compute the standard edge feature vector: ``[rel_pos, dist, rel_vel]``.

    Any component can be disabled with the flags. The returned array has
    shape ``(E, F)``; ``F`` depends on which components are enabled.
    """
    if edges.size == 0:
        return np.zeros((0, 0), dtype=float)
    src = edges[:, 0]
    dst = edges[:, 1]
    feats = []
    rel_pos = positions[dst] - positions[src]
    if include_relative_position:
        feats.append(rel_pos)
    if include_distance:
        feats.append(np.linalg.norm(rel_pos, axis=1, keepdims=True))
    if include_relative_velocity and velocities is not None:
        rel_vel = velocities[dst] - velocities[src]
        feats.append(rel_vel)
    return np.concatenate(feats, axis=1) if feats else np.zeros((edges.shape[0], 0), dtype=float)


def aggregate(
    edge_values: np.ndarray,
    edges: np.ndarray,
    num_nodes: int,
    reduce: str = "mean",
) -> np.ndarray:
    """Aggregate per-edge values into per-node values indexed by ``edges[:, 0]``.

    This is the "source aggregation" used by every policy here — given
    incoming messages per edge, sum/mean/max them at the source node.
    Mirrors PyTorch Geometric's ``scatter`` semantics.
    """
    if edge_values.size == 0:
        out_shape = (num_nodes,) + edge_values.shape[1:]
        return np.zeros(out_shape, dtype=edge_values.dtype)
    src = edges[:, 0]
    out_shape = (num_nodes,) + edge_values.shape[1:]
    if reduce == "sum":
        out = np.zeros(out_shape, dtype=edge_values.dtype)
        np.add.at(out, src, edge_values)
        return out
    if reduce == "mean":
        out = np.zeros(out_shape, dtype=edge_values.dtype)
        counts = np.zeros(num_nodes, dtype=np.int64)
        np.add.at(out, src, edge_values)
        np.add.at(counts, src, 1)
        denom = np.maximum(counts, 1).astype(edge_values.dtype).reshape(-1, *([1] * (edge_values.ndim - 1)))
        return out / denom
    if reduce == "max":
        out = np.full(out_shape, -np.inf, dtype=edge_values.dtype)
        # np.maximum.at exists and is stable for duplicate indices
        np.maximum.at(out, src, edge_values)
        out[np.isinf(out)] = 0.0
        return out
    raise ValueError(f"unknown reduce '{reduce}'; use sum|mean|max")


class MessagePassingPolicy(ABC):
    """Base class for any policy expressed as message passing over the swarm.

    Implementations override :meth:`message`, :meth:`aggregate_messages`
    (optional — defaults to mean aggregation), and :meth:`update`. The
    standard :meth:`step` driver glues the three together. Hand-crafted
    policies supply no parameters; learned policies override the same
    methods in a PyTorch module in ``gossamer.learning``.
    """

    #: Aggregation kernel used by the default :meth:`aggregate_messages`.
    reduce: str = "mean"

    @abstractmethod
    def message(self, graph: InteractionGraph) -> np.ndarray:
        """Compute a per-edge message ``(E, F_msg)`` from graph state."""

    def aggregate_messages(self, messages: np.ndarray, graph: InteractionGraph) -> np.ndarray:
        return aggregate(messages, graph.edges, graph.num_nodes, reduce=self.reduce)

    @abstractmethod
    def update(self, aggregated: np.ndarray, graph: InteractionGraph) -> np.ndarray:
        """Map aggregated messages + graph state to per-node actions ``(N, A)``."""

    def step(self, graph: InteractionGraph) -> np.ndarray:
        messages = self.message(graph)
        agg = self.aggregate_messages(messages, graph)
        return self.update(agg, graph)


__all__ = [
    "InteractionGraph",
    "MessagePassingPolicy",
    "aggregate",
    "build_radius_graph",
    "compute_edge_features",
]
