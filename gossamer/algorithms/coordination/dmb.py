"""
Density-Modulated Boids (DMB).

Fixed-weight Boids oscillates and collides as local density rises: the same
cohesion that forms a flock at low density over-packs it at high density. DMB
makes the three Reynolds weights *functions of local density* via sigmoid
schedules, so alignment and cohesion relax and separation strengthens exactly
where crowding would otherwise trigger the order->disorder transition. This is
the policy whose phase behaviour the DMB + TF-ACO paper characterises; by
construction it reduces to fixed-weight Boids when the schedules are flat.

The module exposes three things:

* :func:`density_modulated_weights` — the pure sigmoid schedules (paper §3.2 /
  Appendix A), trivially unit-testable against the closed form.
* :func:`local_density` — per-agent neighbour count within a radius, computed
  in O(N) via the shared uniform grid.
* :func:`dmb_step` — one integration step: observe (optionally noisy) neighbour
  positions, modulate weights by local density, apply Boids steering, optional
  obstacle repulsion, clamp, and advance.

Previously this logic lived inline in the Maneuver.Map runner; it now lives
here so papers can cite ``gossamer.algorithms.coordination.dmb`` and the runner
only orchestrates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from gossamer.utils.spatial import build_grid, neighbors_within


@dataclass(frozen=True)
class DMBParams:
    """Sigmoid weight-modulation parameters.

    ``d_crit`` is the inflection density (in neighbours within
    ``density_radius``). Alignment and cohesion decay above it; separation
    grows. Setting all ``k_*`` to 0 recovers fixed-weight Boids at the
    schedule midpoints, which is how the paper's fixed-Boids baseline is run
    from the same code path.
    """

    d_crit: float = 12.0
    # Alignment: w_align_max at low density, -> 0 at high density.
    w_align_max: float = 1.0
    k_align: float = 0.5
    # Cohesion: w_coh_max at low density, -> 0 at high density.
    w_coh_max: float = 1.0
    k_coh: float = 0.5
    # Separation: ramps from w_sep_min (low density) to w_sep_max (high density).
    w_sep_min: float = 1.0
    w_sep_max: float = 2.5
    k_sep: float = 0.5
    density_radius: float = 10.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic.
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def density_modulated_weights(
    density: np.ndarray, params: DMBParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map per-agent local density to ``(w_align, w_coh, w_sep)`` arrays.

    Alignment / cohesion use a *decreasing* sigmoid in density; separation
    uses an *increasing* one (paper §3.2; Appendix A gives the same forms).
    """
    d = np.asarray(density, dtype=float)
    w_align = params.w_align_max * _sigmoid(-params.k_align * (d - params.d_crit))
    w_coh = params.w_coh_max * _sigmoid(-params.k_coh * (d - params.d_crit))
    w_sep = params.w_sep_min + (params.w_sep_max - params.w_sep_min) * _sigmoid(
        params.k_sep * (d - params.d_crit)
    )
    return w_align, w_coh, w_sep


def local_density(positions: np.ndarray, radius: float) -> np.ndarray:
    """Per-agent count of other agents within ``radius`` (O(N) via the grid)."""
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    grid, cell_idx = build_grid(positions, radius)
    dens = np.zeros(n, dtype=float)
    for i in range(n):
        dens[i] = len(neighbors_within(positions, cell_idx, grid, radius, i))
    return dens


def dmb_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    params: DMBParams = DMBParams(),
    *,
    neighbor_radius: float = 10.0,
    separation_distance: float = 1.0,
    max_speed: float = 5.0,
    max_accel: Optional[float] = None,
    obstacles: Optional[np.ndarray] = None,
    obstacle_strength: float = 0.0,
    obstacle_range: float = 50.0,
    sensing_noise: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Advance the swarm one step under density-modulated Boids.

    Returns ``(new_positions, new_velocities, info)`` where ``info`` carries
    the per-agent ``density`` and the modulated weight arrays — the runner logs
    these as the ``density`` colour channel, and tests assert against them.

    ``sensing_noise`` (metres, isotropic Gaussian) perturbs the *observed*
    neighbour positions only; bodies still move on their true positions. This
    is the sensing channel of the paper's expanded noise model; the velocity
    and actuator channels are applied by the engine.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if positions.shape != velocities.shape:
        raise ValueError("positions and velocities must have the same shape")
    n, dims = positions.shape

    observed = positions
    if sensing_noise > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        observed = positions + rng.normal(scale=sensing_noise, size=positions.shape)

    cell = max(float(neighbor_radius), float(params.density_radius))
    grid, cell_idx = build_grid(observed, cell)

    dens = np.zeros(n, dtype=float)
    align_vec = np.zeros_like(positions)
    coh_vec = np.zeros_like(positions)
    sep_vec = np.zeros_like(positions)
    nr2 = float(neighbor_radius) ** 2
    dr2 = float(params.density_radius) ** 2
    sr2 = float(separation_distance) ** 2
    eps = 1e-9

    for i in range(n):
        cand = neighbors_within(observed, cell_idx, grid, cell, i)
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)
        diff = observed[cand] - observed[i]
        d2 = np.einsum("ij,ij->i", diff, diff)
        dens[i] = int(np.count_nonzero(d2 <= dr2))
        in_range = d2 <= nr2
        if not np.any(in_range):
            continue
        nbr = cand[in_range]
        diff_r = diff[in_range]
        d2_r = d2[in_range] + eps
        align_vec[i] = velocities[nbr].mean(axis=0) - velocities[i]
        coh_vec[i] = observed[nbr].mean(axis=0) - observed[i]
        close = d2_r < sr2
        if np.any(close):
            sep_vec[i] = np.sum(-diff_r[close] / d2_r[close][:, None], axis=0)

    w_align, w_coh, w_sep = density_modulated_weights(dens, params)
    new_vel = (
        velocities
        + w_align[:, None] * align_vec
        + w_coh[:, None] * coh_vec
        + w_sep[:, None] * sep_vec
    )

    if obstacles is not None and obstacle_strength > 0.0:
        new_vel = new_vel + _obstacle_force(
            positions, np.asarray(obstacles, dtype=float), obstacle_strength, obstacle_range
        )

    # Optional acceleration clamp (reaction-wheel limit), then speed clamp.
    if max_accel is not None:
        dv = new_vel - velocities
        dv_mag = np.linalg.norm(dv, axis=1, keepdims=True)
        scale = np.minimum(1.0, (max_accel * dt) / np.maximum(dv_mag, eps))
        new_vel = velocities + dv * scale
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    over = (speed > max_speed).ravel()
    if np.any(over):
        new_vel[over] = new_vel[over] / speed[over] * max_speed

    new_pos = positions + new_vel * dt
    info = {"density": dens, "w_align": w_align, "w_coh": w_coh, "w_sep": w_sep}
    return new_pos, new_vel, info


def weighted_boids_update(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    w_align,
    w_coh,
    w_sep,
    *,
    neighbor_radius: float = 10.0,
    separation_distance: float = 1.0,
    max_speed: float = 5.0,
    max_accel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Boids update with per-agent (or scalar) rule weights.

    Shared steering core for DMB (sigmoid-scheduled weights) and the
    learned-weight MAPPO baseline (MLP-produced weights). ``w_align``,
    ``w_coh``, ``w_sep`` may each be a scalar or an ``(N,)`` array. With flat
    weights this is plain fixed-weight Boids; the only difference from the
    baseline is *where the weights come from*, which is exactly the comparison
    the DMB paper draws.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    n = positions.shape[0]
    wa = np.broadcast_to(np.asarray(w_align, dtype=float), (n,))
    wc = np.broadcast_to(np.asarray(w_coh, dtype=float), (n,))
    ws = np.broadcast_to(np.asarray(w_sep, dtype=float), (n,))

    cell = float(neighbor_radius)
    grid, cell_idx = build_grid(positions, cell)
    align_vec = np.zeros_like(positions)
    coh_vec = np.zeros_like(positions)
    sep_vec = np.zeros_like(positions)
    sr2 = float(separation_distance) ** 2
    eps = 1e-9

    for i in range(n):
        cand = neighbors_within(positions, cell_idx, grid, cell, i)
        if not cand:
            continue
        cand = np.asarray(cand, dtype=int)  # already within neighbor_radius
        diff = positions[cand] - positions[i]
        d2 = np.einsum("ij,ij->i", diff, diff) + eps
        align_vec[i] = velocities[cand].mean(axis=0) - velocities[i]
        coh_vec[i] = positions[cand].mean(axis=0) - positions[i]
        close = d2 < sr2
        if np.any(close):
            sep_vec[i] = np.sum(-diff[close] / d2[close][:, None], axis=0)

    new_vel = velocities + wa[:, None] * align_vec + wc[:, None] * coh_vec + ws[:, None] * sep_vec
    if max_accel is not None:
        dv = new_vel - velocities
        dv_mag = np.linalg.norm(dv, axis=1, keepdims=True)
        scale = np.minimum(1.0, (max_accel * dt) / np.maximum(dv_mag, eps))
        new_vel = velocities + dv * scale
    speed = np.linalg.norm(new_vel, axis=1, keepdims=True)
    over = (speed > max_speed).ravel()
    if np.any(over):
        new_vel[over] = new_vel[over] / speed[over] * max_speed
    return positions + new_vel * dt, new_vel


def _obstacle_force(
    positions: np.ndarray, obstacles: np.ndarray, strength: float, rng_range: float
) -> np.ndarray:
    """Inverse-square repulsion from a set of obstacle points within range."""
    force = np.zeros_like(positions)
    rng2 = rng_range * rng_range
    for o in obstacles:
        diff = positions - o
        d2 = np.einsum("ij,ij->i", diff, diff) + 1e-9
        mask = d2 < rng2
        if np.any(mask):
            force[mask] += strength * diff[mask] / d2[mask][:, None]
    return force
