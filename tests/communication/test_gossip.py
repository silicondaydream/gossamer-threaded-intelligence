import numpy as np

from gossamer.communication.gossip import GossipProtocol


def test_gossip_push():
    # Simple two-node graph: 0-1
    adj = np.array([[0, 1], [1, 0]], dtype=float)
    proto = GossipProtocol(adj, push=True, random_state=42)
    messages = {0: ['m'], 1: []}
    new = proto.step(messages)
    # Node 1 should receive 'm' due to push gossip
    assert 'm' in new[1]
    # Node 0 retains 'm'
    assert 'm' in new[0]

def test_gossip_pull():
    # Simple two-node graph: 0-1
    adj = np.array([[0, 1], [1, 0]], dtype=float)
    proto = GossipProtocol(adj, push=False, random_state=42)
    messages = {0: ['m'], 1: []}
    new = proto.step(messages)
    # Node 1 may not receive (pull from 0 or other direction)
    # But node 0 should still have 'm'
    assert 'm' in new[0]
    # node 1 pulls: with seed, rng.choice selects neighbor[0]
    assert 'm' in new[1]