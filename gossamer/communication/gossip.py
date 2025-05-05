"""
Gossip protocol: random peer-to-peer message exchange.
"""
import numpy as np
from .base_protocol import BaseProtocol


class GossipProtocol(BaseProtocol):
    """
    Gossip-based communication where agents randomly exchange messages with neighbors.

    Parameters:
        adjacency: adjacency matrix
        push: bool, if True use push gossip, else pull gossip
        random_state: int or None, seed for reproducibility
    """
    def __init__(self, adjacency, push=True, random_state=None):
        super().__init__(adjacency)
        self.push = push
        self.rng = np.random.RandomState(random_state)

    def step(self, messages):  # noqa: D102
        # Initialize with current messages as sets
        new_messages = {i: set(messages.get(i, [])) for i in range(self.n)}
        if self.push:
            # Each agent pushes its messages to a random neighbor
            for i in range(self.n):
                nbrs = self.neighbors[i]
                if not nbrs:
                    continue
                j = self.rng.choice(nbrs)
                new_messages[j].update(messages.get(i, []))
        else:
            # Each agent pulls messages from a random neighbor
            for i in range(self.n):
                nbrs = self.neighbors[i]
                if not nbrs:
                    continue
                j = self.rng.choice(nbrs)
                new_messages[i].update(messages.get(j, []))
        return new_messages