"""
Laplacian average consensus as a zero-parameter message-passing policy.

Classical synchronous average consensus runs ``x_{t+1} = x_t + alpha (A x_t - D x_t)``,
which is a single message-passing step: each edge carries the difference
``x_dst - x_src``, aggregated as a sum per node, then scaled by ``alpha``.

Paired with :class:`~gossamer.graph.InteractionGraph` this lets consensus
be composed with other GNN policies (e.g., run one consensus step between
Boids updates to share SOC or AoI values) without ad-hoc glue code.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from gossamer.graph import InteractionGraph, MessagePassingPolicy


@dataclass
class AverageConsensusGNN(MessagePassingPolicy):
    """One step of average consensus over a node-feature field.

    ``values`` are carried on ``InteractionGraph.node_features``; the class
    is stateless otherwise. ``alpha`` defaults to ``None`` meaning
    ``1/(max_degree+1)`` will be chosen at step time for guaranteed
    stability. Pass an explicit value if you've computed a tighter bound.
    """

    alpha: Optional[float] = None
    reduce: str = "sum"

    def message(self, graph: InteractionGraph) -> np.ndarray:
        if graph.node_features is None:
            raise ValueError("AverageConsensusGNN requires graph.node_features to carry the consensus variable")
        if graph.num_edges == 0:
            return np.zeros((0, graph.node_features.shape[1]), dtype=float)
        src = graph.edges[:, 0]
        dst = graph.edges[:, 1]
        # Signed difference: each neighbor pulls the source toward itself
        return graph.node_features[dst] - graph.node_features[src]

    def update(self, aggregated: np.ndarray, graph: InteractionGraph) -> np.ndarray:
        if graph.node_features is None:
            return aggregated
        alpha = self.alpha
        if alpha is None:
            # Compute max degree from the edge list; Metropolis-style stable choice.
            if graph.num_edges == 0:
                return graph.node_features.copy()
            deg = np.zeros(graph.num_nodes, dtype=np.int64)
            np.add.at(deg, graph.edges[:, 0], 1)
            alpha = 1.0 / float(deg.max() + 1)
        return graph.node_features + alpha * aggregated


__all__ = ["AverageConsensusGNN"]
