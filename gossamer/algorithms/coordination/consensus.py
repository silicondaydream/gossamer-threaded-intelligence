"""
Consensus algorithms for swarm coordination.
"""
import numpy as np

def average_consensus(values, adjacency, alpha=None, iterations=50):
    """
    Perform synchronous average consensus over the network.

    values: array_like, shape (n_agents, k) or (n_agents,)
    adjacency: array_like, shape (n_agents, n_agents), non-negative weights
    alpha: float, step size (if None, set to 1/(max_degree+1))
    iterations: int, number of consensus iterations

    Returns:
        values: ndarray of same shape as input values after consensus
    """
    x = np.asarray(values, dtype=float)
    A = np.asarray(adjacency, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    n = A.shape[0]
    if x.shape[0] != n:
        raise ValueError("values length must match adjacency dimension")

    # Degree vector
    deg = A.sum(axis=1)
    max_deg = deg.max()
    if alpha is None:
        # Stability condition: alpha < 1/(max_degree)
        alpha = 1.0 / (max_deg + 1)

    # Ensure x is 2D
    if x.ndim == 1:
        x = x.reshape(n, 1)

    # Consensus iterations
    for _ in range(iterations):
        # Laplacian term: adjacency @ x - degree * x
        x = x + alpha * (A.dot(x) - deg[:, None] * x)
    return x.squeeze()