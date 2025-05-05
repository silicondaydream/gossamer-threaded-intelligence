"""
Potential field methods for pathfinding and obstacle avoidance.
"""
import numpy as np

def potential_field_force(position, goal, obstacles=None, k_att=1.0, k_rep=100.0, rep_range=10.0):
    """
    Compute the artificial force on an agent at `position` due to an attractive goal and repulsive obstacles.

    Parameters:
        position: array_like, shape (d,)
        goal: array_like, shape (d,)
        obstacles: array_like, shape (n_obs, d)
        k_att: float, attractive gain
        k_rep: float, repulsive gain
        rep_range: float, obstacles influence range

    Returns:
        force: ndarray, shape (d,)
    """
    pos = np.asarray(position, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if pos.ndim != 1 or goal.ndim != 1 or pos.shape != goal.shape:
        raise ValueError("position and goal must be 1D arrays of same dimension")
    d = pos.shape[0]
    # Attractive force toward goal
    F_att = -k_att * (pos - goal)

    # Repulsive force from obstacles
    F_rep = np.zeros(d)
    if obstacles is not None:
        obs = np.asarray(obstacles, dtype=float)
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        if obs.ndim != 2 or obs.shape[1] != d:
            raise ValueError("obstacles must be array of shape (n_obstacles, dims)")
        diff = pos - obs  # shape (n_obs, d)
        dist = np.linalg.norm(diff, axis=1)
        for vec, dist_i in zip(diff, dist):
            if dist_i < rep_range and dist_i > 0:
                # magnitude of repulsive force
                mag = k_rep * (1.0 / dist_i - 1.0 / rep_range) / (dist_i ** 2)
                F_rep += mag * (vec / dist_i)
    return F_att + F_rep

def potential_field_step(position, goal, obstacles=None, step_size=1.0, **kwargs):
    """
    Perform one integration step under potential field dynamics.

    Returns new position.
    """
    force = potential_field_force(position, goal, obstacles, **kwargs)
    norm = np.linalg.norm(force)
    if norm > 0:
        return np.asarray(position, dtype=float) + step_size * force / norm
    return np.asarray(position, dtype=float)