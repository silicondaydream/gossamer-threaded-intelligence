"""
Broadcast protocol: each agent sends its messages to all neighbors.
"""
from .base_protocol import BaseProtocol


class BroadcastProtocol(BaseProtocol):
    """
    Broadcast messages to all neighbors in the communication graph.
    """
    def step(self, messages):  # noqa: D102
        # Initialize with current messages as sets
        new_messages = {i: set(messages.get(i, [])) for i in range(self.n)}
        # Propagate messages to neighbors
        for i in range(self.n):
            msgs = messages.get(i, [])
            for j in self.neighbors[i]:
                new_messages[j].update(msgs)
        return new_messages