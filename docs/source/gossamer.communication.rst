gossamer.communication package
==============================

This package implements communication protocols for decentralized multi-agent
systems. Efficient information exchange is crucial for coordinated behaviors,
and these protocols define how agents share messages across network topologies.

Key protocols:
- `BaseProtocol`: Abstract interface for synchronous communication steps
- Broadcast: Flood messages to all reachable neighbors
- Gossip: Randomized peer-to-peer message exchange

Combine or extend protocols to model varying levels of network connectivity,
reliability, and bandwidth constraints.

.. automodule:: gossamer.communication.base_protocol
   :members:
   :undoc-members:

.. automodule:: gossamer.communication.broadcast
   :members:
   :undoc-members:

.. automodule:: gossamer.communication.gossip
   :members:
   :undoc-members: