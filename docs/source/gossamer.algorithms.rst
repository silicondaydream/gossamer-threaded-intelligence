gossamer.algorithms package
===========================

The `gossamer.algorithms` package provides core swarm intelligence
algorithms for coordination, navigation, optimization, and resilience.
Each subpackage implements a family of strategies to enable agents to
collectively explore, optimize, and recover within dynamic environments.

Subpackages:
- Coordination: Consensus protocols, Boids-style flocking, and task
  allocation strategies
- Navigation: Potential field-based path planning and exploration
- Optimization: Standard swarm optimizers (e.g., PSO)
- Resilience: Self-healing network topologies and fault-tolerant logic

Use these building blocks to compose complex multi-agent behaviors
and analyze emergent phenomena.

Coordination Subpackage
-----------------------

.. automodule:: gossamer.algorithms.coordination.consensus
   :members:
   :undoc-members:

.. automodule:: gossamer.algorithms.coordination.flocking
   :members:
   :undoc-members:

.. automodule:: gossamer.algorithms.coordination.task_allocation
   :members:
   :undoc-members:

Navigation Subpackage
---------------------

.. automodule:: gossamer.algorithms.navigation.potential_field
   :members:
   :undoc-members:

Optimization Subpackage
-----------------------

.. automodule:: gossamer.algorithms.optimization.pso
   :members:
   :undoc-members:

Resilience Subpackage
---------------------

.. automodule:: gossamer.algorithms.resilience.self_healing_topology
   :members:
   :undoc-members: