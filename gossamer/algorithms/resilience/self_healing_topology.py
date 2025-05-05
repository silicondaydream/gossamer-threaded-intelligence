"""
Fault tolerance and self-healing topology management.
"""
import numpy as np
import networkx as nx

def self_healing_topology(adjacency, failed_nodes):
    """
    Given an adjacency matrix and a list of failed node indices, produce a healed adjacency
    matrix by reconnecting the neighbors of each failed node to each other.

    Parameters:
        adjacency: array_like, shape (n, n), binary or weight matrix
        failed_nodes: list of int indices of failed nodes

    Returns:
        healed_adj: ndarray, shape (n, n)
    """
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n = A.shape[0]
    G = nx.from_numpy_array(A)
    # Process each failed node
    for f in failed_nodes:
        if f not in G:
            continue
        neighbors = list(G.neighbors(f))
        # Fully connect neighbors
        for i, u in enumerate(neighbors):
            for v in neighbors[i+1:]:
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
        # Remove failed node
        G.remove_node(f)
    # Build healed adjacency
    healed = np.zeros((n, n), dtype=float)
    for u, v, data in G.edges(data=True):
        healed[u, v] = data.get('weight', 1.0)
        healed[v, u] = healed[u, v]
    # Ensure failed nodes have zero rows/cols
    for f in failed_nodes:
        if 0 <= f < n:
            healed[f, :] = 0.0
            healed[:, f] = 0.0
    return healed