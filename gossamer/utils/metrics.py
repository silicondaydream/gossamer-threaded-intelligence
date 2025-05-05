"""Functions to calculate swarm metrics such as order and cohesion."""

import numpy as np

def cohesion(positions):
    """Calculate cohesion as average distance of agents to the centroid."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array of shape (n_agents, n_dims)")
    centroid = positions.mean(axis=0)
    distances = np.linalg.norm(positions - centroid, axis=1)
    return distances.mean()