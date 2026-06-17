"""
Learned baselines the swarm papers compare against.

Two policies, both 2-layer MLPs over per-agent *local* features (the locality
lives in the features — density, neighbour variance, centrality — so an MLP
captures it without bespoke message passing):

* :class:`LearnedBoidsWeights` — the DMB paper's MAPPO learned-weight Boids
  baseline. Maps local density / neighbour-velocity variance to the three
  Reynolds weights. The paper's claim is that DMB's hand-designed sigmoids
  match this learned policy's order parameter at zero training cost; this
  module is the policy that makes that an apples-to-apples comparison.
* :class:`RelaySelectionPolicy` — the ICCD paper's MAPPO relay-selection
  baseline. Maps state-of-charge / degree / density to a relay logit, trained
  against reward = +Delta-freshness - lambda * energy.

Reward functions and the action->control mappings that plug these into the
NumPy algorithms live here too, so a trainer (``gossamer.learning.mappo`` or an
external one) only has to supply the optimisation loop.

PyTorch is required (this lives under ``gossamer.learning``); the NumPy
algorithms in ``gossamer.algorithms`` do not import it.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from gossamer.algorithms.coordination.dmb import local_density, weighted_boids_update


# --------------------------------------------------------------------------
# DMB: learned-weight Boids
# --------------------------------------------------------------------------

class LearnedBoidsWeights(nn.Module):
    """MLP mapping per-agent features -> positive ``(w_align, w_coh, w_sep)``.

    Softplus on the output keeps weights non-negative, matching the domain of
    the hand-designed DMB schedules so the two are directly comparable.
    """

    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softplus(self.net(features))


def boids_features(positions: np.ndarray, velocities: np.ndarray, radius: float) -> np.ndarray:
    """Per-agent ``[local_density, neighbour_speed_variance, own_speed]``.

    These are the local statistics the DMB paper says the learned policy
    conditions on. Returns an ``(N, 3)`` float array.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    n = positions.shape[0]
    dens = local_density(positions, radius)
    from gossamer.utils.spatial import build_grid, neighbors_within
    grid, cell_idx = build_grid(positions, radius)
    nbr_var = np.zeros(n)
    speeds = np.linalg.norm(velocities, axis=1)
    for i in range(n):
        nbrs = neighbors_within(positions, cell_idx, grid, radius, i)
        if nbrs:
            nbr_var[i] = float(np.var(speeds[np.asarray(nbrs, dtype=int)]))
    return np.stack([dens, nbr_var, speeds], axis=1)


def learned_boids_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    policy: LearnedBoidsWeights,
    *,
    feature_radius: float = 10.0,
    neighbor_radius: float = 10.0,
    separation_distance: float = 1.0,
    max_speed: float = 5.0,
    max_accel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """One step of Boids whose weights come from ``policy`` instead of sigmoids."""
    feats = boids_features(positions, velocities, feature_radius)
    with torch.no_grad():
        w = policy(torch.as_tensor(feats, dtype=torch.float32)).cpu().numpy()
    new_pos, new_vel = weighted_boids_update(
        positions, velocities, dt, w[:, 0], w[:, 1], w[:, 2],
        neighbor_radius=neighbor_radius, separation_distance=separation_distance,
        max_speed=max_speed, max_accel=max_accel,
    )
    return new_pos, new_vel, {"w_align": w[:, 0], "w_coh": w[:, 1], "w_sep": w[:, 2]}


def boids_reward(psi: float, collision_rate: float, lam: float = 1.0) -> float:
    """DMB training reward: ``+psi - lambda * collision_rate`` (paper §4.4)."""
    return float(psi) - lam * float(collision_rate)


# --------------------------------------------------------------------------
# ICCD: learned relay selection
# --------------------------------------------------------------------------

class RelaySelectionPolicy(nn.Module):
    """MLP mapping per-agent ``[soc, degree, density]`` -> a relay logit."""

    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def relay_features(soc: np.ndarray, degree: np.ndarray, density: np.ndarray) -> np.ndarray:
    """Stack ``[soc, degree, density]`` into an ``(N, 3)`` feature array."""
    return np.stack([
        np.asarray(soc, dtype=float),
        np.asarray(degree, dtype=float),
        np.asarray(density, dtype=float),
    ], axis=1)


def select_relays_learned(
    logits: np.ndarray, fraction: float = 0.05, threshold: Optional[float] = None
) -> np.ndarray:
    """Boolean relay mask from policy logits.

    With ``threshold`` set, relays are agents whose logit exceeds it; otherwise
    the top ``fraction`` of agents by logit are chosen (at least one).
    """
    logits = np.asarray(logits, dtype=float)
    n = logits.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n == 0:
        return mask
    if threshold is not None:
        return logits > threshold
    k = max(1, int(round(fraction * n)))
    top = np.argsort(logits)[-k:]
    mask[top] = True
    return mask


def relay_reward(delta_freshness: float, energy: float, lam: float = 1.0) -> float:
    """ICCD relay reward: ``+Delta-freshness - lambda * energy`` (paper §4.4)."""
    return float(delta_freshness) - lam * float(energy)


__all__ = [
    "LearnedBoidsWeights",
    "RelaySelectionPolicy",
    "boids_features",
    "boids_reward",
    "learned_boids_step",
    "relay_features",
    "relay_reward",
    "select_relays_learned",
]
