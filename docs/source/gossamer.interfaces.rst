gossamer.interfaces package
===========================

Adapter modules for integrating Gossamer with external simulation engines or
tools. These interfaces provide a unified API so that agents and algorithms
can operate within diverse environments without changing core logic.

Current adapters:
- LeviathanInterface: Wraps a Leviathan environment to expose Gossamer’s
  `step`, `metrics`, and `run` methods.

Future adapters can support additional simulators (e.g., Maneuver.Map backend,
ROS, custom frameworks) by implementing the same interface.

.. automodule:: gossamer.interfaces.leviathan_interface
   :members:
   :undoc-members: