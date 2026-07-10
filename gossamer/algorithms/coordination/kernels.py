"""Vectorised, edge-list coordination kernels — the single implementation.

Every coordination primitive existed twice. ``gossamer...primitive.py`` held the
canonical, unit-tested registry, built on per-agent Python loops (``flocking.py``
walks ``for i in range(n_agents)``). ``maneuver-map/backend/app/policies.py`` held
``_PRIMITIVE_DISPATCH``, a separate vectorised reimplementation that the runner
actually executed. The two were kept in agreement by tests and a doc comment.

The fork had a real cause: the per-agent loops cannot run the N=128k ladder. The
fix is not to delete the fast versions but to *move them here*, so the fast path
is the only path and Maneuver.Map imports rather than reimplements. Orrery adds at
least three more primitives; each would otherwise fork the same way.

These are byte-for-byte the kernels that produced the shipped P1/P3 grids —
moved, not rewritten. Reduction order is preserved (``np.add.at`` over the same
edge ordering from ``cKDTree.query_pairs``), because floating-point summation is
not associative and a reordering would perturb published numbers.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["kdtree_edges", "boids_accel_edges", "as_col"]


def kdtree_edges(pos: np.ndarray, radius: float) -> Tuple[np.ndarray, np.ndarray]:
    """Undirected neighbour pairs ``(i < j)`` within ``radius``, as index arrays.

    Built at C speed by a KD-tree. Shared by every primitive's movement /
    interaction graph, replacing the per-agent neighbour loops.
    """
    n = pos.shape[0]
    if n > 1:
        pairs = cKDTree(pos).query_pairs(radius, output_type="ndarray")
    else:
        pairs = np.empty((0, 2), dtype=np.intp)
    return pairs[:, 0], pairs[:, 1]


def as_col(w):
    """Broadcast a scalar or per-agent ``(N,)`` weight against ``(N, 3)`` vectors."""
    a = np.asarray(w, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else float(a)


def boids_accel_edges(pos, vel, eu, ev, dt, w_align, w_coh, w_sep,
                      sep_d, max_speed) -> np.ndarray:
    """Vectorised Boids steering from a neighbour edge list.

    O(edges), no per-agent Python loop, no N×N matrices. Weights may be scalars
    (fixed-weight flocking) or per-agent ``(N,)`` arrays (DMB's density-modulated
    schedule). Semantics match ``weighted_boids_update``: alignment is the mean
    neighbour velocity minus own, cohesion the mean neighbour position minus own,
    separation ``Σ -diff/d²`` over neighbours closer than ``sep_d``.

    Returns the acceleration that realises the new velocity over ``dt``.
    """
    n = pos.shape[0]
    deg = np.zeros(n)
    sum_vel = np.zeros_like(vel)
    sum_pos = np.zeros_like(pos)
    sep = np.zeros_like(pos)
    if eu.size:
        np.add.at(deg, eu, 1.0)
        np.add.at(deg, ev, 1.0)
        np.add.at(sum_vel, eu, vel[ev])
        np.add.at(sum_vel, ev, vel[eu])
        np.add.at(sum_pos, eu, pos[ev])
        np.add.at(sum_pos, ev, pos[eu])
        d = pos[eu] - pos[ev]
        dist2 = np.einsum("ij,ij->i", d, d) + 1e-9
        close = dist2 < sep_d * sep_d
        if np.any(close):
            rep = np.zeros_like(d)
            rep[close] = d[close] / dist2[close][:, None]
            np.add.at(sep, eu, rep)
            np.add.at(sep, ev, -rep)
    has = deg > 0
    align = np.zeros_like(vel)
    coh = np.zeros_like(pos)
    if np.any(has):
        align[has] = sum_vel[has] / deg[has, None] - vel[has]
        coh[has] = sum_pos[has] / deg[has, None] - pos[has]
    new_vel = vel + as_col(w_align) * align + as_col(w_coh) * coh + as_col(w_sep) * sep
    spd = np.linalg.norm(new_vel, axis=1)
    fast = spd > max_speed
    if np.any(fast):
        new_vel[fast] = new_vel[fast] / spd[fast, None] * max_speed
    return (new_vel - vel) / max(dt, 1e-9)
