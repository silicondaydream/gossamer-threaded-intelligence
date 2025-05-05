"""
SwarmSimulator: orchestrates simulation of agent swarms using coordination algorithms.
"""
import numpy as np

from gossamer.algorithms.coordination.flocking import flock_step
from gossamer.utils.metrics import cohesion, alignment, separation


class SwarmSimulator:
    """
    Simulator for swarm agent dynamics (Boids flocking by default).

    Parameters:
        n_agents: int, number of agents (ignored if positions provided)
        n_dims: int, spatial dimensions (ignored if positions provided)
        dt: float, timestep size
        alignment_weight: float, weight for alignment force
        cohesion_weight: float, weight for cohesion force
        separation_weight: float, weight for separation force
        neighbor_radius: float, neighbor detection radius
        separation_distance: float, min distance for separation
        max_speed: float, maximum speed per agent
        positions: array_like, optional initial positions (n_agents x n_dims)
        velocities: array_like, optional initial velocities (n_agents x n_dims)
    """
    def __init__(
        self,
        n_agents=None,
        n_dims=2,
        dt=0.1,
        alignment_weight=1.0,
        cohesion_weight=1.0,
        separation_weight=1.5,
        neighbor_radius=10.0,
        separation_distance=1.0,
        max_speed=5.0,
        positions=None,
        velocities=None,
    ):
        # Initialize or validate positions
        if positions is not None:
            self.positions = np.asarray(positions, dtype=float)
            if self.positions.ndim != 2:
                raise ValueError("positions must be 2D array")
            self.n_agents, self.n_dims = self.positions.shape
        else:
            if n_agents is None:
                raise ValueError("n_agents must be provided if positions not set")
            self.n_agents = n_agents
            self.n_dims = n_dims
            self.positions = np.random.rand(self.n_agents, self.n_dims) * 100.0
        # Initialize or validate velocities
        if velocities is not None:
            self.velocities = np.asarray(velocities, dtype=float)
            if self.velocities.shape != (self.n_agents, self.n_dims):
                raise ValueError("velocities shape must match positions")
        else:
            # Random unit velocities
            angles = np.random.rand(self.n_agents) * 2 * np.pi
            if self.n_dims == 2:
                self.velocities = np.vstack((np.cos(angles), np.sin(angles))).T
            else:
                # For higher dims, sample normal and normalize
                v = np.random.randn(self.n_agents, self.n_dims)
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                self.velocities = v / np.where(norm > 0, norm, 1)
        # Simulation parameters
        self.dt = dt
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight
        self.neighbor_radius = neighbor_radius
        self.separation_distance = separation_distance
        self.max_speed = max_speed

    def step(self):  # noqa: C901
        """Advance simulation by one timestep using flocking dynamics."""
        self.positions, self.velocities = flock_step(
            self.positions,
            self.velocities,
            self.dt,
            alignment_weight=self.alignment_weight,
            cohesion_weight=self.cohesion_weight,
            separation_weight=self.separation_weight,
            neighbor_radius=self.neighbor_radius,
            separation_distance=self.separation_distance,
            max_speed=self.max_speed,
        )
        return self.positions, self.velocities

    def metrics(self):
        """Compute and return key swarm metrics: cohesion, alignment, separation."""
        return {
            'cohesion': cohesion(self.positions),
            'alignment': alignment(self.velocities),
            'separation': separation(self.positions),
        }

    def run(self, steps, callback=None):
        """
        Run the simulation for a given number of steps.

        callback: optional function(step, positions, velocities, metrics)
        """
        for step in range(steps):
            self.step()
            if callback:
                callback(step, self.positions, self.velocities, self.metrics())
        return self.positions, self.velocities