import numpy as np
import pytest

from gossamer.utils.metrics import cohesion, alignment, separation


def test_cohesion_simple():
    pos = np.array([[0.0, 0.0], [2.0, 0.0]])
    # Centroid at (1,0), distances are 1.0
    assert pytest.approx(cohesion(pos), rel=1e-6) == 1.0


def test_cohesion_invalid():
    with pytest.raises(ValueError):
        cohesion([1, 2, 3])


def test_alignment_perfect():
    vel = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert pytest.approx(alignment(vel), rel=1e-6) == 1.0


def test_alignment_opposite():
    vel = np.array([[1.0, 0.0], [-1.0, 0.0]])
    # Unit vectors cancel out
    assert pytest.approx(alignment(vel), abs=1e-6) == 0.0


def test_alignment_invalid():
    with pytest.raises(ValueError):
        alignment([1, 2, 3])


def test_separation_simple():
    pos = np.array([[0.0, 0.0], [3.0, 4.0]])
    # Distance is 5.0
    assert pytest.approx(separation(pos), rel=1e-6) == 5.0


def test_separation_single_agent():
    pos = np.array([[0.0, 0.0]])
    # No neighbors => zero by definition
    assert separation(pos) == 0.0


def test_separation_invalid():
    with pytest.raises(ValueError):
        separation([1, 2, 3])