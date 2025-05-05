#!/usr/bin/env python3
"""
Run a simple flocking (Boids) simulation demo.
"""
import numpy as np

from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.utils.metrics import cohesion, alignment, separation


def main():
    # Simulation parameters
    n_agents = 20
    n_dims = 2
    steps = 50
    dt = 0.1
    # Initialize positions and velocities
    positions = np.random.rand(n_agents, n_dims) * 100.0
    angles = np.random.rand(n_agents) * 2 * np.pi
    velocities = np.vstack((np.cos(angles), np.sin(angles))).T

    for step in range(steps):
        positions, velocities = flock_step(positions, velocities, dt)
        centroid = positions.mean(axis=0)
        coh = cohesion(positions)
        order = alignment(velocities)
        sep = separation(positions)
        print(
            f"Step {step}: centroid={centroid}, cohesion={coh:.3f}, "
            f"alignment={order:.3f}, separation={sep:.3f}"
        )


if __name__ == "__main__":
    main()