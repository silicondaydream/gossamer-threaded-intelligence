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
 
def alignment(velocities):
    """Alignment metric: magnitude of average normalized velocity units."""
    velocities = np.asarray(velocities, dtype=float)
    if velocities.ndim != 2:
        raise ValueError("velocities must be a 2D array of shape (n_agents, n_dims)")
    # Normalize velocities to unit vectors, handle zero speeds
    speeds = np.linalg.norm(velocities, axis=1, keepdims=True)
    unit_vel = np.divide(
        velocities,
        speeds,
        where=speeds > 0,
        out=np.zeros_like(velocities),
    )
    # Sum unit vectors and compute magnitude per agent
    sum_unit = unit_vel.sum(axis=0)
    order = np.linalg.norm(sum_unit) / velocities.shape[0]
    return order

def separation(positions):
    """Separation metric: average nearest-neighbor distance."""
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array of shape (n_agents, n_dims)")
    # Compute pairwise distances
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    # Ignore self-distance by setting to infinity
    n = positions.shape[0]
    if n < 2:
        return 0.0
    idx = np.arange(n)
    distances[idx, idx] = np.inf
    # Nearest neighbor distance per agent
    min_dist = distances.min(axis=1)
    return min_dist.mean()