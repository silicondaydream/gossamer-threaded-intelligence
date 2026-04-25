"""
Gossip protocol: random peer-to-peer message exchange.
"""
from typing import Optional, Union

import numpy as np
from .base_protocol import BaseProtocol


class GossipProtocol(BaseProtocol):
    """
    Gossip-based communication where agents randomly exchange messages with neighbors.

    Parameters:
        adjacency: adjacency matrix
        push: bool, if True use push gossip, else pull gossip
        rng: an ``np.random.Generator`` or int seed (None for a nondeterministic
            default). The legacy ``random_state`` keyword is accepted as a seed
            for backwards compatibility.
    """
    def __init__(
        self,
        adjacency,
        push: bool = True,
        rng: Optional[Union[int, np.random.Generator]] = None,
        random_state: Optional[int] = None,
    ):
        super().__init__(adjacency)
        self.push = push
        if isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            seed = rng if rng is not None else random_state
            self.rng = np.random.default_rng(seed)

    def step(self, messages):  # noqa: D102
        # Initialize with current messages as sets
        new_messages = {i: set(messages.get(i, [])) for i in range(self.n)}
        if self.push:
            # Each agent pushes its messages to a random neighbor
            for i in range(self.n):
                nbrs = self.neighbors[i]
                if not nbrs:
                    continue
                j = int(self.rng.choice(nbrs))
                new_messages[j].update(messages.get(i, []))
        else:
            # Each agent pulls messages from a random neighbor
            for i in range(self.n):
                nbrs = self.neighbors[i]
                if not nbrs:
                    continue
                j = int(self.rng.choice(nbrs))
                new_messages[i].update(messages.get(j, []))
        return new_messages