import numpy as np

from gossamer.communication.broadcast import BroadcastProtocol


def test_broadcast_simple():
    # Line graph: 0-1-2
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)
    proto = BroadcastProtocol(adj)
    # Initial messages: only node 0 has 'a'
    messages = {0: ['a']}
    new = proto.step(messages)
    # Node 0 retains 'a'
    assert 'a' in new[0]
    # Node 1 receives 'a'
    assert 'a' in new[1]
    # Node 2 should not receive (distance 2)
    assert 'a' not in new[2]

def test_broadcast_multiple_msgs():
    adj = np.ones((2, 2), dtype=float)
    proto = BroadcastProtocol(adj)
    messages = {0: ['x'], 1: ['y']}
    new = proto.step(messages)
    # Fully connected: each gets both
    assert set(new[0]) == {'x', 'y'}
    assert set(new[1]) == {'x', 'y'}