"""
Criticality and phase-transition instruments.

When the DMB / TF-ACO papers claim a "supercritical density threshold",
reviewers want to see the usual suspects:

* An **order parameter** ``psi`` as a function of the control parameter
  (density, noise, etc.).
* **Susceptibility** ``chi = N * (<psi^2> - <psi>^2)`` peaking at the
  transition.
* **Binder cumulant** ``U = 1 - <psi^4> / (3 <psi^2>^2)`` for locating
  ``T_c`` by curve crossings across sizes.
* **Correlation length** from the velocity correlation function.
* **Branching ratio** / avalanche statistics for self-organized
  criticality claims.

This module gives you those numbers from a time-series of snapshots.
Nothing here is novel; it's the standard active-matter / statistical-
mechanics toolkit applied to whatever observable the experiment
produces.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def susceptibility(order_parameter_series: Sequence[float], n_agents: int) -> float:
    """``chi = N * Var(psi)`` computed over a time series at fixed control point.

    ``n_agents`` is used to make ``chi`` intensive-per-agent-scaled so
    comparisons across system sizes are meaningful.
    """
    psi = np.asarray(order_parameter_series, dtype=float)
    if psi.size < 2:
        return 0.0
    return float(n_agents * np.var(psi, ddof=1))


def binder_cumulant(order_parameter_series: Sequence[float]) -> float:
    """``U = 1 - <psi^4> / (3 <psi^2>^2)``.

    Sized curves of U vs control parameter cross near the critical point
    (finite-size scaling). A single number here; sweep across system
    sizes to use it properly.
    """
    psi = np.asarray(order_parameter_series, dtype=float)
    if psi.size < 2:
        return 0.0
    m2 = float(np.mean(psi ** 2))
    m4 = float(np.mean(psi ** 4))
    if m2 <= 0:
        return 0.0
    return float(1.0 - m4 / (3.0 * m2 ** 2))


def velocity_correlation(velocities: np.ndarray, positions: np.ndarray,
                         n_bins: int = 32, max_r: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """Radial velocity correlation ``C(r) = <dv_i . dv_j>_{|r_i - r_j| ~ r}``.

    ``dv = v - <v>`` is the fluctuation from the mean velocity. Returns
    ``(r_edges, C)`` where C is the average correlation in each radial bin.
    Useful for extracting correlation length by locating where C(r)
    crosses zero.
    """
    v = np.asarray(velocities, dtype=float)
    p = np.asarray(positions, dtype=float)
    n = v.shape[0]
    if n < 2:
        return np.zeros(n_bins + 1), np.zeros(n_bins)
    dv = v - v.mean(axis=0, keepdims=True)
    # Pairwise distances and dot products (O(N^2) memory — use small N here)
    diff = p[:, None, :] - p[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    ij = np.triu_indices(n, k=1)
    dots = np.einsum("ij,ij->i", dv[ij[0]], dv[ij[1]])
    dists = d[ij]
    if max_r is None:
        max_r = float(dists.max()) if dists.size else 1.0
    bins = np.linspace(0.0, max_r, n_bins + 1)
    which = np.clip(np.digitize(dists, bins) - 1, 0, n_bins - 1)
    c = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=np.int64)
    np.add.at(c, which, dots)
    np.add.at(counts, which, 1)
    counts = np.maximum(counts, 1)
    c = c / counts
    return bins, c


def correlation_length(r_edges: np.ndarray, c: np.ndarray) -> float:
    """First zero-crossing radius of ``C(r)``; a common proxy for xi.

    Returns ``inf`` if the correlation never crosses zero in the measured
    window — a signature of super-critical long-range order.
    """
    if c.size == 0:
        return 0.0
    centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    sign_change = np.where((c[:-1] > 0) & (c[1:] <= 0))[0]
    if sign_change.size == 0:
        return float("inf")
    i = int(sign_change[0])
    # Linear interpolation for the crossing
    c0, c1 = c[i], c[i + 1]
    r0, r1 = centers[i], centers[i + 1]
    if c0 == c1:
        return float(r0)
    t = c0 / (c0 - c1)
    return float(r0 + t * (r1 - r0))


def branching_ratio(event_counts: Sequence[int]) -> float:
    """Mean branching ratio ``sigma = <n_{t+1}> / <n_t>`` from an event train.

    ``sigma == 1`` indicates a critical process (self-organized
    criticality for neural avalanches, bundle-storm cascades, etc.);
    ``< 1`` subcritical, ``> 1`` supercritical.
    """
    x = np.asarray(event_counts, dtype=float)
    if x.size < 2:
        return 0.0
    num = float(np.mean(x[1:]))
    den = float(np.mean(x[:-1]))
    return num / max(den, 1e-12)


def avalanche_size_distribution(event_counts: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sizes, counts)`` for avalanches defined as runs of non-zero ``event_counts``.

    An avalanche starts when ``event_counts`` transitions from 0 to >0 and
    ends when it returns to 0; its "size" is the sum of events in between.
    Standard SOC instrument; plot on log-log and fit a power-law exponent
    to report the universality class.
    """
    x = np.asarray(event_counts, dtype=int)
    sizes = []
    current = 0
    in_avalanche = False
    for v in x:
        if v > 0:
            current += int(v)
            in_avalanche = True
        else:
            if in_avalanche:
                sizes.append(current)
                current = 0
                in_avalanche = False
    if in_avalanche and current > 0:
        sizes.append(current)
    if not sizes:
        return np.array([], dtype=int), np.array([], dtype=int)
    vals, counts = np.unique(np.asarray(sizes, dtype=int), return_counts=True)
    return vals, counts


__all__ = [
    "avalanche_size_distribution",
    "binder_cumulant",
    "branching_ratio",
    "correlation_length",
    "susceptibility",
    "velocity_correlation",
]
