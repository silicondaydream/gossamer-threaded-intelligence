import numpy as np
import pytest

from gossamer.algorithms.coordination.consensus import average_consensus


def test_average_consensus_basic():
    # Simple line graph of 3 nodes
    adj = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ], dtype=float)
    values = np.array([0.0, 10.0, 20.0])
    result = average_consensus(values, adj, alpha=0.2, iterations=100)
    # Consensus value should be the average = 10
    assert np.allclose(result, 10.0, atol=1e-2)


def test_average_consensus_invalid():
    with pytest.raises(ValueError):
        average_consensus([1, 2, 3], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        average_consensus(np.zeros((3,)), np.zeros((4, 4)))