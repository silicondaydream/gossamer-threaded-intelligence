Swarm Simulator
===============

The `SwarmSimulator` class provides a flexible environment to simulate
flocking (Boids-style) dynamics for a collection of agents.
It handles initialization of agent positions and velocities, computes
alignment, cohesion, and separation forces, and integrates trajectories
over discrete timesteps.

Key features:
  - Configurable number of agents and spatial dimensions
  - Adjustable weights for alignment, cohesion, and separation behaviors
  - Range-based neighbor detection and separation thresholds
  - Convenient callback hooks for logging or visualization

Use this simulator to experiment with emergent flocking behaviors or
to integrate with custom agent models by subclassing or wrapping.

.. automodule:: gossamer.simulator
   :members:
   :undoc-members:
   :show-inheritance: