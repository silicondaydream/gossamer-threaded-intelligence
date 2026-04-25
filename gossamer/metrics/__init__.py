"""
Expanded metric suite for swarm research.

The legacy metrics (cohesion, alignment, separation) live in
``gossamer.utils.metrics``. This package adds:

* ``gossamer.metrics.info`` — information-theoretic measures (mutual
  information, transfer entropy) for per-agent influence analysis.
* ``gossamer.metrics.criticality`` — phase-transition instruments
  (susceptibility, Binder cumulant, branching ratio, correlation length).
* ``gossamer.metrics.graph`` — interaction-graph spectral and structural
  measures (algebraic connectivity, spectral gap, degree distribution,
  mean clustering coefficient).

These unblock the phase-transition paper rewrite and make it possible to
report the instruments reviewers expect (critical exponents, universality
class indicators) instead of raw percent-improvement claims.
"""

from gossamer.metrics import criticality, graph, info

__all__ = ["info", "criticality", "graph"]
