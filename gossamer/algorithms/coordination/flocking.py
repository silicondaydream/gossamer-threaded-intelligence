"""
Flocking algorithm (Boids) implementation for swarm coordination.
"""
from itertools import product
from typing import Optional
import numpy as np

from gossamer.utils.spatial import build_grid as _build_spatial_grid  # re-exported for back-compat


def _flock_step_spatial(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    alignment_weight: float,
    cohesion_weight: float,
    separation_weight: float,
    neighbor_radius: float,
    separation_distance: float,
    max_speed: float,
):
    n_agents, n_dims = positions.shape
    grid, idx = _build_spatial_grid(positions, neighbor_radius)
    offsets = list(product([-1, 0, 1], repeat=n_dims))
    new_velocities = np.zeros_like(velocities)
    nr = float(neighbor_radius)
    sr = float(separation_distance)
    nr2 = nr * nr
    sr2 = sr * sr
    eps = 1e-9

    for i in range(n_agents):
        key = idx[i]
        neighbors = []
        for off in offsets:
            cell = tuple(key[d] + off[d] for d in range(n_dims))
            neighbors.extend(grid.get(cell, []))
        if neighbors:
            nbr_idx = np.array([j for j in neighbors if j != i], dtype=int)
        else:
            nbr_idx = np.array([], dtype=int)
        v = velocities[i].copy()
        if nbr_idx.size:
            diff = positions[nbr_idx] - positions[i]
            d2 = np.einsum("ij,ij->i", diff, diff)
            in_range = d2 < nr2
            if np.any(in_range):
                nbr_idx = nbr_idx[in_range]
                diff = diff[in_range]
                d2 = d2[in_range] + eps
                # Alignment
                avg_vel = velocities[nbr_idx].mean(axis=0)
                align = alignment_weight * (avg_vel - velocities[i])
                # Cohesion
                center = positions[nbr_idx].mean(axis=0)
                coh = cohesion_weight * (center - positions[i])
                # Separation
                close = d2 < sr2
                sep = np.zeros(n_dims)
                if np.any(close):
                    vectors = -diff[close]
                    sep = separation_weight * np.sum(vectors / d2[close][:, None], axis=0)
                v = velocities[i] + align + coh + sep
        speed = np.linalg.norm(v)
        if speed > max_speed:
            v = v / speed * max_speed
        new_velocities[i] = v
    new_positions = positions + new_velocities * dt
    return new_positions, new_velocities


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
    use_spatial: Optional[bool] = None,
    spatial_threshold: int = 2000,
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
    use_spatial: bool or None, use spatial hashing for neighbor search (auto if None)
    spatial_threshold: int, agent count threshold to enable spatial hashing in auto mode

    Returns:
        new_positions: ndarray, updated positions
        new_velocities: ndarray, updated velocities
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    n_agents, n_dims = positions.shape
    if use_spatial is None:
        use_spatial = n_agents > spatial_threshold
    if use_spatial:
        return _flock_step_spatial(
            positions,
            velocities,
            dt,
            alignment_weight,
            cohesion_weight,
            separation_weight,
            neighbor_radius,
            separation_distance,
            max_speed,
        )

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
