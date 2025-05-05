"""
Voting mechanisms for distributed decision-making.
"""

import numpy as np
from typing import List, Any, Dict, Tuple, Optional

__all__ = [
    "get_candidates",
    "plurality_voting",
    "borda_count",
    "approval_voting",
    "pairwise_preferences",
    "condorcet_winner",
    "schulze_method",
]

def get_candidates(ballots: List[List[Any]]) -> List[Any]:
    """
    Extract unique candidates from ballots, preserving order of first appearance.
    """
    candidates: List[Any] = []
    seen = set()
    for ballot in ballots:
        for c in ballot:
            if c not in seen:
                seen.add(c)
                candidates.append(c)
    return candidates

def plurality_voting(ballots: List[List[Any]]) -> Tuple[List[Any], Dict[Any, int]]:
    """
    Plurality voting: each ballot votes for its top-ranked candidate.
    Returns a tuple of (winners, vote counts).
    """
    candidates = get_candidates(ballots)
    counts: Dict[Any, int] = {c: 0 for c in candidates}
    for ballot in ballots:
        if ballot:
            counts[ballot[0]] += 1
    if counts:
        max_votes = max(counts.values())
        winners = [c for c, v in counts.items() if v == max_votes]
    else:
        winners = []
    return winners, counts

def borda_count(ballots: List[List[Any]]) -> Tuple[List[Any], Dict[Any, int]]:
    """
    Borda count voting: assigns points based on ranking positions.
    Returns a tuple of (winners, scores).
    """
    candidates = get_candidates(ballots)
    m = len(candidates)
    scores: Dict[Any, int] = {c: 0 for c in candidates}
    for ballot in ballots:
        for idx, c in enumerate(ballot):
            # Highest-ranked candidate gets (m - 1) points
            scores[c] += m - idx - 1
    if scores:
        max_score = max(scores.values())
        winners = [c for c, s in scores.items() if s == max_score]
    else:
        winners = []
    return winners, scores

def approval_voting(ballots: List[List[Any]]) -> Tuple[List[Any], Dict[Any, int]]:
    """
    Approval voting: each ballot approves one or more candidates.
    Returns a tuple of (winners, approval counts).
    """
    candidates = get_candidates(ballots)
    counts: Dict[Any, int] = {c: 0 for c in candidates}
    for ballot in ballots:
        for c in ballot:
            counts[c] = counts.get(c, 0) + 1
    if counts:
        max_votes = max(counts.values())
        winners = [c for c, v in counts.items() if v == max_votes]
    else:
        winners = []
    return winners, counts

def pairwise_preferences(
    ballots: List[List[Any]], candidates: Optional[List[Any]] = None
) -> np.ndarray:
    """
    Build pairwise preference matrix P where P[i][j] is
    the number of ballots preferring candidate i over j.
    """
    if candidates is None:
        candidates = get_candidates(ballots)
    n = len(candidates)
    # Map candidate to index
    index: Dict[Any, int] = {c: i for i, c in enumerate(candidates)}
    P = np.zeros((n, n), dtype=int)
    for ballot in ballots:
        # position (rank) of each candidate in the ballot
        rank: Dict[Any, int] = {c: r for r, c in enumerate(ballot)}
        for ci in candidates:
            i = index[ci]
            for cj in candidates:
                j = index[cj]
                if i == j:
                    continue
                ri = rank.get(ci, None)
                rj = rank.get(cj, None)
                if ri is not None and rj is not None:
                    if ri < rj:
                        P[i, j] += 1
                elif ri is not None:
                    P[i, j] += 1
                # else: no preference
    return P

def condorcet_winner(ballots: List[List[Any]]) -> List[Any]:
    """
    Identify Condorcet winner(s): candidates that beat every other
    candidate in pairwise comparisons. Returns list of winners (possibly empty).
    """
    candidates = get_candidates(ballots)
    if not candidates:
        return []
    P = pairwise_preferences(ballots, candidates)
    n = len(candidates)
    winners: List[Any] = []
    for i, ci in enumerate(candidates):
        if all(P[i, j] > P[j, i] for j in range(n) if j != i):
            winners.append(ci)
    return winners

def schulze_method(ballots: List[List[Any]]) -> Tuple[List[Any], np.ndarray]:
    """
    Schulze method for Condorcet elections.
    Returns a tuple of (winners, strongest path matrix).
    """
    candidates = get_candidates(ballots)
    n = len(candidates)
    if n == 0:
        return [], np.zeros((0, 0), dtype=int)
    # Pairwise preferences
    P = pairwise_preferences(ballots, candidates)
    # Initialize strongest paths matrix
    p = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j:
                if P[i, j] > P[j, i]:
                    p[i, j] = P[i, j]
                else:
                    p[i, j] = 0
    # Compute strongest paths
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for k in range(n):
                if i == k or j == k:
                    continue
                p[j, k] = max(p[j, k], min(p[j, i], p[i, k]))
    # Determine winners
    winners: List[Any] = []
    for i, ci in enumerate(candidates):
        if all(p[i, j] >= p[j, i] for j in range(n) if j != i):
            winners.append(ci)
    return winners, p