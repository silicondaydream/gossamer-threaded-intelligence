import numpy as np
import pytest

from gossamer.communication.base_protocol import BaseProtocol


class DummyProtocol(BaseProtocol):
    def step(self, messages):
        return messages


def test_base_protocol_invalid_adj():
    with pytest.raises(ValueError):
        DummyProtocol([1, 2, 3])

def test_base_protocol_neighbors():
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)
    proto = DummyProtocol(adj)
    # neighbors mapping
    assert proto.neighbors[0] == [1]
    assert proto.neighbors[1] == [0, 2]
    assert proto.neighbors[2] == [1]