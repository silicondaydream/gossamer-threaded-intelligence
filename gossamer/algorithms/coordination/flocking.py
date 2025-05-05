"""
Flocking algorithm (Boids) implementation for swarm coordination.
"""
import numpy as np

def flock_step(
    positions,
    velocities,
    dt,
    alignment_weight=1.0,
    cohesion_weight=1.0,
    separation_weight=1.5,
    neighbor_radius=10.0,
    separation_distance=1.0,
    max_speed=5.0,
):
    """
    Perform one timestep update for a flock of agents.

    positions: array_like, shape (n_agents, n_dims)
    velocities: array_like, shape (n_agents, n_dims)
    dt: float, timestep
    alignment_weight: weight for velocity matching
    cohesion_weight: weight for center-of-mass attraction
    separation_weight: weight for collision avoidance
    neighbor_radius: float, radius to consider neighbors
    separation_distance: float, distance threshold for separation
    max_speed: float, maximum allowed speed

    Returns:
        new_positions: ndarray, updated positions
        new_velocities: ndarray, updated velocities
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    n_agents, n_dims = positions.shape

    # Precompute pairwise differences and distances
    diff = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(diff, axis=2)

    new_velocities = np.zeros_like(velocities)
    for i in range(n_agents):
        # Identify neighbors (excluding self)
        neighbor_idx = [j for j in range(n_agents) if j != i and distances[i, j] < neighbor_radius]
        v = velocities[i].copy()
        if neighbor_idx:
            # Alignment: match average neighbor velocity
            avg_vel = velocities[neighbor_idx].mean(axis=0)
            align = alignment_weight * (avg_vel - velocities[i])
            # Cohesion: move toward neighbors' center of mass
            center = positions[neighbor_idx].mean(axis=0)
            coh = cohesion_weight * (center - positions[i])
            # Separation: avoid crowding
            close_idx = [j for j in neighbor_idx if distances[i, j] < separation_distance]
            sep = np.zeros(n_dims)
            if close_idx:
                vectors = diff[i, close_idx, :]
                dist_close = distances[i, close_idx][:, None]
                sep = separation_weight * np.sum(vectors / (dist_close ** 2), axis=0)
            # Sum contributions
            v = velocities[i] + align + coh + sep
        # Limit speed
        speed = np.linalg.norm(v)
        if speed > max_speed:
            v = v / speed * max_speed
        new_velocities[i] = v
    # Update positions
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities