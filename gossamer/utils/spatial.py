"""
Spatial neighborhood primitives shared across Gossamer algorithms.

Flocking, potential fields, ICCD contact detection, HMA market proximity,
and TF-ACO heuristics all need the same question answered: *for each agent,
which other agents lie within radius R?* Doing it naively is O(N^2) in time
and memory; at N=1e6 that blows the machine.

This module provides a single uniform-grid implementation:

* :func:`build_grid(positions, cell_size)` places each agent in a voxel
  indexed by floor(position / cell_size) and returns a dict of
  voxel -> list of agent indices, plus the per-agent voxel index array.
* :func:`neighbors_within` enumerates candidates in the agent's voxel and
  its 3x3x3 neighborhood and filters to the true radius.

``cell_size`` should be set equal to (or slightly larger than) the query
radius; otherwise the 3x3x3 neighborhood stops being a conservative
bound. For adaptive-radius callers, use the largest radius you'll query.
"""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Tuple

import numpy as np


GridIndex = Dict[Tuple[int, ...], List[int]]


def build_grid(positions: np.ndarray, cell_size: float) -> Tuple[GridIndex, np.ndarray]:
    """Build a uniform-grid spatial index over agent positions.

    Parameters
    ----------
    positions:
        Array of shape ``(N, D)`` where D is 2 or 3.
    cell_size:
        Edge length of each grid voxel. Use ``max(query_radius, 1e-9)``.

    Returns
    -------
    grid:
        Mapping from voxel index tuple ``(ix, iy[, iz])`` to the list of
        agent indices that fall in that voxel.
    cell_idx:
        Integer array of shape ``(N, D)`` giving each agent's voxel index.
        Returned so callers can reuse it without re-dividing.
    """
    if positions.size == 0:
        return {}, np.zeros((0, positions.shape[1] if positions.ndim == 2 else 1), dtype=int)
    cell = max(float(cell_size), 1e-9)
    cell_idx = np.floor(positions / cell).astype(int)
    grid: GridIndex = {}
    for i, key in enumerate(map(tuple, cell_idx)):
        grid.setdefault(key, []).append(i)
    return grid, cell_idx


def neighbors_within(
    positions: np.ndarray,
    cell_idx: np.ndarray,
    grid: GridIndex,
    radius: float,
    agent: int,
) -> List[int]:
    """Return indices of neighbors of ``agent`` strictly within ``radius``.

    Self is excluded. Callers that want to include self should add it back.

    Uses a 3x3x3 neighborhood over voxels, so ``grid`` must have been built
    with a cell size >= ``radius``. At smaller cells this will under-count;
    at much larger cells it still works but degrades to O(N^2) in the worst
    case.
    """
    if grid is None or cell_idx.size == 0:
        return []
    r = max(float(radius), 1e-9)
    key = tuple(cell_idx[agent])
    d = len(key)
    candidates: List[int] = []
    for offset in product((-1, 0, 1), repeat=d):
        voxel = tuple(key[k] + offset[k] for k in range(d))
        if voxel in grid:
            candidates.extend(grid[voxel])
    if not candidates:
        return []
    diff = positions[candidates] - positions[agent]
    d2 = np.einsum("ij,ij->i", diff, diff)
    r2 = r * r
    return [candidates[j] for j in range(len(candidates)) if d2[j] <= r2 and candidates[j] != agent]
