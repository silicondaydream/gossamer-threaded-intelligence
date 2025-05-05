gossamer.environment package
============================

The `gossamer.environment` package defines perception models that simulate
how agents sense their surroundings. Different models capture sensing
capabilities and limitations such as range, field-of-view, noise,
latency, and packet loss.

Available perception classes:
- PerfectPerception: omnidirectional, noise-free (optional Gaussian noise)
- RangePerception: detect only within a specified radius
- FieldOfViewPerception: cone-based sensing relative to agent heading
- DelayPerception & StochasticDelayPerception: introduce observation latency
- PacketLossPerception: random neighbor drop
- BearingOnlyPerception: only angular information
- IntermittentBlindSpotPerception: random occlusion sectors

Use these models to simulate realistic or constrained sensing in agent-based
simulations.

.. automodule:: gossamer.environment.perception
   :members:
   :undoc-members: