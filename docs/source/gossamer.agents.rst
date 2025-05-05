gossamer.agents package
========================

This package defines the base classes and interfaces for agent behaviors
within the Gossamer ecosystem. Agents encapsulate state, perception, and
decision-making logic, and can be customized or extended for specific
simulation needs.

Key components:
  - `BaseAgent`: Abstract interface for agent implementations
  - Simple or task-oriented agent examples can subclass `BaseAgent`

Extend this package by creating new agent classes that implement
the core lifecycle methods (`sense`, `decide`, `act`).

.. automodule:: gossamer.agents.base_agent
   :members:
   :undoc-members: