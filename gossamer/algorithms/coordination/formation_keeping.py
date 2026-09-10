"""Separation-keeping policies for a formation-flying cluster (N4).

The seam mirrors `orbital_market`: the *world* (Hill/CW dynamics, propellant,
conjunction screening, STL monitoring) lives in Leviathan and the campaign that
drives it lives in Maneuver.Map. What lives here is the decision — given what a
satellite believes about the formation, what Delta-v does it command.

A policy is ``(states, t, ctx) -> (N, 3) commanded Delta-v`` in LVLH m/s. It
returns commands, never applies them: the engine's `Propulsion.apply` decides
what was actually delivered, and the difference between the two is the whole
point of N4's cost axis.

`open_loop` is the do-nothing floor and is not decoration. Without it a policy
that "maintains the formation" cannot be shown to be doing anything, and a
configuration whose envelope holds unaided reads as a control success.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["FormationPolicy", "FORMATION_POLICIES", "PolicyContext",
           "conjunction_avoidance_policy", "open_loop_policy",
           "station_keeping_policy"]


@dataclass(frozen=True)
class PolicyContext:
    """What a policy is allowed to know. Mean motion is the chief's, rad/s."""

    mean_motion: float
    d_safe_m: float
    control_period_s: float
    dt_s: float


#: ``(positions, velocities, t, ctx) -> (N, 3)`` commanded Delta-v [m/s].
FormationPolicy = Callable[[np.ndarray, np.ndarray, float, PolicyContext], np.ndarray]


def open_loop_policy(pos: np.ndarray, vel: np.ndarray, t: float,
                     ctx: PolicyContext) -> np.ndarray:
    """Never burn. The floor every other policy must beat to have done anything."""
    return np.zeros_like(pos)


def station_keeping_policy(pos: np.ndarray, vel: np.ndarray, t: float,
                           ctx: PolicyContext) -> np.ndarray:
    """Restore the closed-orbit condition ``vy = -2 n x`` on the control period.

    A CW relative orbit is closed exactly when this holds; otherwise along-track
    separation grows secularly at ``-3 (vy + 2 n x)``. This nulls that term and
    nothing else — it does not target a formation shape, so it holds whatever
    geometry it inherits, drift and all.
    """
    dv = np.zeros_like(pos)
    if not _on_control_step(t, ctx):
        return dv
    dv[:, 1] = -2.0 * ctx.mean_motion * pos[:, 0] - vel[:, 1]
    return dv


def conjunction_avoidance_policy(pos: np.ndarray, vel: np.ndarray, t: float,
                                 ctx: PolicyContext,
                                 trigger_scale: float = 2.0,
                                 dv_m_s: float = 0.02) -> np.ndarray:
    """Burn radially apart when a pair is inside ``trigger_scale * d_safe``.

    Reactive and deliberately unsophisticated: it is the baseline a planned
    manoeuvre has to beat, and its failure mode (burning late, or burning into
    a third satellite) is the reason a planner is worth writing.

    Radial rather than along-track on purpose: a radial impulse changes the
    relative ellipse's size, while an along-track one mostly changes phase and
    comes back around a quarter orbit later.
    """
    dv = np.zeros_like(pos)
    if not _on_control_step(t, ctx):
        return dv

    trigger = trigger_scale * ctx.d_safe_m
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            sep = pos[j] - pos[i]
            dist = float(np.linalg.norm(sep))
            if dist >= trigger:
                continue
            if dist <= 0.0:
                push = np.array([1.0, 0.0, 0.0])
            else:
                push = sep / dist
            dv[i] -= dv_m_s * push
            dv[j] += dv_m_s * push
    return dv


def _on_control_step(t: float, ctx: PolicyContext) -> bool:
    """True on the samples a controller is allowed to act on.

    A policy that burns every sample is a different experiment from one that
    burns every control period, and letting `dt_s` decide which is which makes
    the Delta-v bill a function of the sampling grid.
    """
    if ctx.control_period_s <= 0.0:
        return True
    phase = (t + 0.5 * ctx.dt_s) % ctx.control_period_s
    return phase < ctx.dt_s


FORMATION_POLICIES: Dict[str, FormationPolicy] = {
    "open_loop": open_loop_policy,
    "station_keeping": station_keeping_policy,
    "conjunction_avoidance": conjunction_avoidance_policy,
}
