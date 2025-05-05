import numpy as np
import pytest

from gossamer.decision_making.voting_mechanisms import (
    get_candidates,
    plurality_voting,
    borda_count,
    approval_voting,
    pairwise_preferences,
    condorcet_winner,
    schulze_method,
)

def test_get_candidates():
    ballots = [['A', 'B'], ['B', 'C', 'A'], ['D'], []]
    candidates = get_candidates(ballots)
    assert candidates == ['A', 'B', 'C', 'D']

def test_plurality_voting():
    ballots = [['A', 'B'], ['A'], ['B', 'A'], []]
    winners, counts = plurality_voting(ballots)
    assert counts == {'A': 2, 'B': 1}
    assert winners == ['A']

def test_borda_count():
    ballots = [['A', 'B', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B']]
    winners, scores = borda_count(ballots)
    assert set(winners) == {'A', 'B', 'C'}
    assert scores == {'A': 3, 'B': 3, 'C': 3}

def test_approval_voting():
    ballots = [['A', 'B'], ['A'], ['B', 'C'], []]
    winners, counts = approval_voting(ballots)
    assert counts == {'A': 2, 'B': 2, 'C': 1}
    assert set(winners) == {'A', 'B'}

def test_pairwise_preferences_full_rank():
    ballots = [['A', 'B', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B']]
    P = pairwise_preferences(ballots)
    # Should be a 3x3 matrix
    assert isinstance(P, np.ndarray)
    assert P.shape == (3, 3)
    # Diagonal should be zeros
    assert np.all(np.diag(P) == 0)
    # For full rankings, P[i,j] + P[j,i] == total ballots
    total = len(ballots)
    for i in range(3):
        for j in range(3):
            if i != j:
                assert P[i, j] + P[j, i] == total

def test_condorcet_winner_two_candidates():
    ballots = [['A', 'B'], ['A'], ['B', 'A']]
    winners = condorcet_winner(ballots)
    assert winners == ['A']

def test_condorcet_winner_none():
    ballots = [['A', 'B'], ['B', 'A']]
    winners = condorcet_winner(ballots)
    assert winners == []

def test_schulze_method_simple():
    ballots = [['A', 'B'], ['A', 'B'], ['B', 'A']]
    winners, p = schulze_method(ballots)
    # A wins
    assert winners == ['A']
    # p should be a numpy array with shape (2,2)
    assert isinstance(p, np.ndarray)
    assert p.shape == (2, 2)