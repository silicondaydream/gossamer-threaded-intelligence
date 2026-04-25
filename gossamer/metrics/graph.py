"""
Spectral and structural metrics for the interaction graph.

The swarm's instantaneous topology controls how information propagates:
algebraic connectivity bounds consensus speed, spectral gap predicts
synchronization, degree distribution distinguishes random vs scale-free
structures, and the clustering coefficient measures local redundancy.
Computing these each step is cheap (O(N) for degree, O(N k) for
dominant eigenvalues via Lanczos) and tells you more about the system
than positions alone.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from gossamer.graph import InteractionGraph


def adjacency_matrix(graph: InteractionGraph, symmetric: bool = True, weighted: bool = False) -> np.ndarray:
    """Build an (N, N) adjacency matrix from the edge list.

    ``symmetric``: treat the edge list as undirected (``A[i, j] = A[j, i]``).
    ``weighted``: use ``1 / (distance + 1)`` as the weight if edge features
    are present; otherwise 1.
    """
    n = graph.num_nodes
    A = np.zeros((n, n), dtype=float)
    if graph.edges.size == 0:
        return A
    src = graph.edges[:, 0]
    dst = graph.edges[:, 1]
    if weighted and graph.edge_features is not None and graph.edge_features.shape[1] > 0:
        # Heuristic: last feature is distance when produced by
        # compute_edge_features with include_distance=True
        dist = graph.edge_features[:, -1]
        w = 1.0 / (dist + 1.0)
    else:
        w = np.ones(graph.edges.shape[0])
    A[src, dst] = w
    if symmetric:
        A[dst, src] = w
    return A


def degree_distribution(graph: InteractionGraph) -> np.ndarray:
    """Per-node out-degree as an ``(N,)`` int array."""
    if graph.edges.size == 0:
        return np.zeros(graph.num_nodes, dtype=np.int64)
    deg = np.zeros(graph.num_nodes, dtype=np.int64)
    np.add.at(deg, graph.edges[:, 0], 1)
    return deg


def algebraic_connectivity(graph: InteractionGraph) -> float:
    """Fiedler value: second-smallest eigenvalue of the Laplacian.

    Lower bound on consensus mixing rate; zero iff the graph is
    disconnected. ``O(N^3)`` via dense eigensolve; swap in scipy's
    ``eigsh(which="SM")`` for large N.
    """
    A = adjacency_matrix(graph, symmetric=True)
    n = A.shape[0]
    if n < 2:
        return 0.0
    D = np.diag(A.sum(axis=1))
    L = D - A
    w = np.linalg.eigvalsh(L)
    w.sort()
    return float(max(w[1], 0.0))


def spectral_gap(graph: InteractionGraph) -> float:
    """Difference between the two largest adjacency eigenvalues.

    Large gap => strong community structure / synchronization. Useful
    together with algebraic_connectivity to characterize regime.
    """
    A = adjacency_matrix(graph, symmetric=True)
    if A.shape[0] < 2:
        return 0.0
    w = np.linalg.eigvalsh(A)
    w.sort()
    return float(w[-1] - w[-2])


def clustering_coefficient(graph: InteractionGraph) -> float:
    """Mean local clustering coefficient averaged over nodes.

    For each node, count how many of its neighbors are mutually
    connected; ratio against the maximum possible gives the local
    coefficient; then average. ``O(N d^2)`` where d is mean degree; fine
    for up to ~10^4 agents at moderate density.
    """
    if graph.num_edges == 0:
        return 0.0
    n = graph.num_nodes
    # Build per-node neighbor sets
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for src, dst in graph.edges:
        if src != dst:
            neighbors[int(src)].add(int(dst))
            neighbors[int(dst)].add(int(src))
    total = 0.0
    counted = 0
    for i, nbrs in enumerate(neighbors):
        k = len(nbrs)
        if k < 2:
            continue
        # Count edges among neighbors
        connections = 0
        nlist = list(nbrs)
        for a in range(len(nlist)):
            for b in range(a + 1, len(nlist)):
                if nlist[b] in neighbors[nlist[a]]:
                    connections += 1
        total += (2.0 * connections) / (k * (k - 1))
        counted += 1
    return float(total / max(counted, 1))


def summary(graph: InteractionGraph) -> dict:
    """Single-call structural snapshot for a paper's results table."""
    deg = degree_distribution(graph)
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.num_edges),
        "mean_degree": float(deg.mean()) if deg.size else 0.0,
        "max_degree": int(deg.max()) if deg.size else 0,
        "algebraic_connectivity": algebraic_connectivity(graph),
        "spectral_gap": spectral_gap(graph),
        "clustering_coefficient": clustering_coefficient(graph),
    }


__all__ = [
    "adjacency_matrix",
    "algebraic_connectivity",
    "clustering_coefficient",
    "degree_distribution",
    "spectral_gap",
    "summary",
]
