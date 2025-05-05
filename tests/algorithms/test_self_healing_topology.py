import numpy as np
import pytest

from gossamer.algorithms.resilience.self_healing_topology import self_healing_topology


def test_self_healing_simple_chain():
    # Chain graph: 0-1-2-3
    adj = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0]
    ], dtype=float)
    failed = [1]
    healed = self_healing_topology(adj, failed)
    # Node 1 should be removed (row and col zero)
    assert np.all(healed[1, :] == 0)
    assert np.all(healed[:, 1] == 0)
    # 0 and 2 should now be directly connected
    assert healed[0, 2] == 1.0
    assert healed[2, 0] == 1.0
    # Other edges preserved: 2-3 remains
    assert healed[2, 3] == 1.0
    assert healed[3, 2] == 1.0

def test_self_healing_no_failures():
    adj = np.eye(3)
    healed = self_healing_topology(adj, [])
    assert np.allclose(healed, adj)

def test_self_healing_invalid_adj():
    with pytest.raises(ValueError):
        self_healing_topology([1, 2, 3], [0])