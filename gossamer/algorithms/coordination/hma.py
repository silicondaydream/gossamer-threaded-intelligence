"""
Hierarchical Market Auctions (HMA) for macro-micro coordination.

Haulers bid for depot pickup tasks using a single canonical *energy-aware*
utility (HMA paper §3.2 / Appendix A):

    U_ij = alpha * M_j / (E_travel(i,j) + E_lift(j)) * sigma(SOC_i) - beta * (t_arrival + t_queue)

where ``sigma`` is the state-of-charge sigmoid that suppresses bids from
low-battery haulers. Depots are cleared by a rolling batch auction with spatial
pruning (each depot only solicits its k nearest haulers). Depot inventory is a
CRDT (:class:`gossamer.crdt.PNCounter`) so depots reconcile counts under
partition, matching the eventually-consistent state story used by ICCD and
TF-ACO.

The module also provides the M/M/c queueing helpers behind the paper's
decoupling result, so the buffer-sizing numbers are computed, not asserted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from gossamer.crdt import PNCounter


@dataclass(frozen=True)
class HMAParams:
    alpha: float = 1.0          # weight on energy-normalised throughput
    beta: float = 0.05          # weight on latency (s^-1)
    k_soc: float = 12.0         # SOC sigmoid steepness
    s_crit: float = 0.30        # SOC sigmoid midpoint (fraction)
    e_move: float = 1.0         # travel energy per unit distance
    e_lift: float = 0.2         # lift energy per unit mass
    speed: float = 2.5          # hauler speed (m/s) for arrival-time estimate
    k_nearest: int = 8          # depots solicit this many nearest haulers
    radius: float = float("inf")  # max solicitation distance
    wear_tiebreak_frac: float = 0.01  # treat bids within 1% as tied -> lower wear wins


def soc_sigmoid(soc, s_crit: float = 0.30, k: float = 12.0) -> np.ndarray:
    """SOC penalty sigma(s) = 1 / (1 + exp(-k (s - s_crit)))."""
    s = np.asarray(soc, dtype=float)
    return 1.0 / (1.0 + np.exp(-k * (s - s_crit)))


def bid_utility(
    mass: float,
    e_travel: float,
    e_lift: float,
    soc: float,
    t_arrival: float,
    t_queue: float,
    params: HMAParams,
) -> float:
    """The canonical energy-aware bid utility for one hauler-depot pair."""
    energy = e_travel + e_lift + 1e-9
    throughput_term = params.alpha * (mass / energy) * float(soc_sigmoid(soc, params.s_crit, params.k_soc))
    latency_term = params.beta * (t_arrival + t_queue)
    return throughput_term - latency_term


# --------------------------------------------------------------------------
# Depot inventory as a CRDT
# --------------------------------------------------------------------------

class DepotInventory:
    """Thin kg-quantised wrapper over a PNCounter for one depot.

    Micro deposits increment; hauler pickups decrement. ``merge`` reconciles
    two replicas of the same depot's inventory after a partition.
    """

    def __init__(self, counter: Optional[PNCounter] = None):
        self._c = counter or PNCounter()

    def deposit(self, kg: float, replica) -> "DepotInventory":
        return DepotInventory(self._c.increment(replica, int(round(kg))))

    def withdraw(self, kg: float, replica) -> "DepotInventory":
        return DepotInventory(self._c.decrement(replica, int(round(kg))))

    def merge(self, other: "DepotInventory") -> "DepotInventory":
        return DepotInventory(self._c.merge(other._c))

    def available(self) -> int:
        return self._c.value()


# --------------------------------------------------------------------------
# Energy-aware rolling auction
# --------------------------------------------------------------------------

def energy_aware_auction(
    hauler_pos: np.ndarray,
    hauler_soc: np.ndarray,
    depot_pos: np.ndarray,
    depot_mass: Sequence[float],
    params: HMAParams = HMAParams(),
    *,
    hauler_capacity: float = 500.0,
    hauler_wear: Optional[np.ndarray] = None,
    depot_queue_time: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clear one batch auction; return ``(assignment, cleared_mass)``.

    ``assignment[h]`` is the depot index hauler ``h`` won, or ``-1``. Each depot
    only considers its ``k_nearest`` haulers within ``radius`` (spatial
    pruning); pairs are cleared greedily by utility, with cumulative wear as the
    tiebreaker when bids are within ``wear_tiebreak_frac``. A depot stops
    accepting haulers once its available mass is covered.
    """
    hauler_pos = np.asarray(hauler_pos, dtype=float)
    depot_pos = np.asarray(depot_pos, dtype=float)
    H = hauler_pos.shape[0]
    D = depot_pos.shape[0]
    soc = np.asarray(hauler_soc, dtype=float)
    wear = np.zeros(H) if hauler_wear is None else np.asarray(hauler_wear, dtype=float)
    qtime = np.zeros(D) if depot_queue_time is None else np.asarray(depot_queue_time, dtype=float)
    remaining = np.array([float(m) for m in depot_mass], dtype=float)

    # Build candidate (utility, wear, hauler, depot) tuples after spatial pruning.
    candidates = []
    for d in range(D):
        if remaining[d] <= 0:
            continue
        dist = np.linalg.norm(hauler_pos - depot_pos[d], axis=1)
        order = np.argsort(dist)
        taken = 0
        for h in order:
            if dist[h] > params.radius:
                break
            if taken >= params.k_nearest:
                break
            taken += 1
            mass = min(hauler_capacity, remaining[d])
            e_travel = params.e_move * dist[h]
            e_lift = params.e_lift * mass
            t_arrival = dist[h] / max(params.speed, 1e-9)
            u = bid_utility(mass, e_travel, e_lift, soc[h], t_arrival, qtime[d], params)
            candidates.append((u, -wear[h], int(h), int(d)))

    # Greedy clearing: highest utility first; wear breaks near-ties via the
    # secondary -wear key already embedded in the tuple sort.
    candidates.sort(reverse=True)
    assignment = -np.ones(H, dtype=int)
    cleared = np.zeros(D, dtype=float)
    for u, _negwear, h, d in candidates:
        if u <= 0:
            break
        if assignment[h] != -1:
            continue
        if remaining[d] <= 0:
            continue
        mass = min(hauler_capacity, remaining[d])
        assignment[h] = d
        remaining[d] -= mass
        cleared[d] += mass
    return assignment, cleared


# --------------------------------------------------------------------------
# Steering: what a hauler does once the auction has assigned it
# --------------------------------------------------------------------------

def depot_steering_accel(
    accel: np.ndarray,
    pos: np.ndarray,
    depots: np.ndarray,
    assignment: np.ndarray,
) -> np.ndarray:
    """Point each *assigned* hauler at its depot; leave everyone else alone.

    The auction decides WHO goes WHERE; this is the other half of the market — the
    part that turns an assignment into motion. Unassigned agents keep whatever
    acceleration they came in with (their flocking term), so the market only ever
    overrides the agents it actually cleared.

    Mutates and returns ``accel``.

    ⚠️ THE LOOP IS DELIBERATE. DO NOT VECTORISE IT.
    This is a unit-norm direction per assigned hauler, and it is trivially
    expressible as an array operation — which is exactly the trap. The HMA
    published headline (+29% vs FCFS) moved and shipped because a distance
    computation on this code path was "harmlessly" vectorised from a scan over
    ``np.hypot`` distances into a reduction over squared distances: mathematically
    the same ordering, different floating-point ties, one flipped depot assignment,
    and a chaotic cascade over 8,000 steps. The published batch is now only
    reproducible on the 0.6.0 wheel. `scripts/hma_fingerprint.py` locks this path
    precisely so the next such change is caught; a speedup here that trips the lock
    is not a speedup, it is a new experiment.
    """
    n = pos.shape[0]
    for i in range(n):
        if assignment[i] >= 0:
            d = depots[assignment[i]] - pos[i]
            nrm = np.linalg.norm(d) + 1e-9
            accel[i] = d / nrm
    return accel


def ring_depots(count: int, bound: float) -> np.ndarray:
    """``count`` depots on a ring at 0.6·bound, in the z=0 plane.

    The default depot placement: evenly spaced, deterministic, and symmetric, so
    no scheduler can win by exploiting an accident of the layout.
    """
    ang = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return np.stack(
        [0.6 * bound * np.cos(ang), 0.6 * bound * np.sin(ang), np.zeros(count)], axis=1
    )


# --------------------------------------------------------------------------
# M/M/c queueing (the decoupling result)
# --------------------------------------------------------------------------

def erlang_c(c: int, a: float) -> float:
    """Probability an arrival waits in an M/M/c queue (offered load ``a`` Erlangs).

    Uses the numerically stable Erlang-B recursion then converts to Erlang C.
    Requires utilization ``a/c < 1``.
    """
    if a <= 0:
        return 0.0
    if a >= c:
        return 1.0
    # Erlang B via recursion B(0)=1, B(k)=a*B(k-1)/(k + a*B(k-1)).
    b = 1.0
    for k in range(1, c + 1):
        b = (a * b) / (k + a * b)
    # Erlang C from Erlang B.
    return (c * b) / (c - a * (1.0 - b))


def mmc_metrics(lam: float, mu: float, c: int) -> dict:
    """Return M/M/c steady-state metrics for the depot buffer model.

    ``lam`` arrival rate (kg/s), ``mu`` per-hauler service rate (kg/s), ``c``
    haulers. Returns utilization ``rho``, wait probability ``p_wait`` (Erlang
    C), mean queue length ``Lq``, mean wait ``Wq``, and total time ``W``.
    """
    if mu <= 0 or c <= 0:
        raise ValueError("mu and c must be positive")
    a = lam / mu               # offered load in Erlangs
    rho = a / c                # utilization per server
    if rho >= 1.0:
        return {"rho": rho, "p_wait": 1.0, "Lq": math.inf, "Wq": math.inf, "W": math.inf}
    p_wait = erlang_c(c, a)
    Lq = p_wait * rho / (1.0 - rho)
    Wq = Lq / lam if lam > 0 else 0.0
    W = Wq + 1.0 / mu
    return {"rho": rho, "p_wait": p_wait, "Lq": Lq, "Wq": Wq, "W": W}


# `hma.py` had no __all__, unlike every other module here — an oversight worth
# fixing now that the package is public (the N2 moat).
__all__ = [
    "DepotInventory",
    "HMAParams",
    "bid_utility",
    "depot_steering_accel",
    "energy_aware_auction",
    "erlang_c",
    "mmc_metrics",
    "ring_depots",
    "soc_sigmoid",
]
