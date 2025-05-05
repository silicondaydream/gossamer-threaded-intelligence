"""
Particle Swarm Optimization (PSO) implementation.
"""
import numpy as np

def pso(objective, bounds, n_particles=30, max_iter=100, w=0.5, c1=1.5, c2=1.5):
    """
    Perform PSO to minimize the given objective function.

    Parameters:
        objective: callable, f(x) -> float, x is 1D array of length d
        bounds: sequence of (min, max) pairs for each dimension (length d)
        n_particles: int, number of particles
        max_iter: int, number of iterations
        w: inertia weight
        c1: cognitive weight
        c2: social weight

    Returns:
        best_pos: ndarray, best-found position
        best_val: float, objective(best_pos)
    """
    bounds = np.array(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must be a sequence of (min, max) pairs")
    d = bounds.shape[0]

    # Initialize particle positions and velocities
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    pos = lb + (ub - lb) * np.random.rand(n_particles, d)
    vel = np.zeros((n_particles, d))

    # Personal bests
    pbest_pos = pos.copy()
    pbest_val = np.array([objective(x) for x in pos])
    # Global best
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    for _ in range(max_iter):
        r1 = np.random.rand(n_particles, d)
        r2 = np.random.rand(n_particles, d)
        # Update velocities
        vel = (
            w * vel
            + c1 * r1 * (pbest_pos - pos)
            + c2 * r2 * (gbest_pos - pos)
        )
        # Update positions
        pos = pos + vel
        # Clamp positions within bounds
        pos = np.clip(pos, lb, ub)

        # Evaluate
        vals = np.array([objective(x) for x in pos])
        # Update personal bests
        better = vals < pbest_val
        pbest_pos[better] = pos[better]
        pbest_val[better] = vals[better]
        # Update global best
        idx = np.argmin(pbest_val)
        if pbest_val[idx] < gbest_val:
            gbest_val = pbest_val[idx]
            gbest_pos = pbest_pos[idx].copy()

    return gbest_pos, float(gbest_val)