"""The HMA market as a stateful world, not an inlined loop in the runner.

`hma.py` shipped the *decisions* — `energy_aware_auction`, `DepotInventory` (a
PNCounter CRDT), `mmc_metrics` (M/M/c + Little's Law) — and Maneuver.Map's runner
consumed exactly one of them (`energy_aware_auction`) while hand-rolling everything
else inline, across ~120 lines of its stepping loop. Three consequences:

  * **The CRDT was dead code.** Its only caller mutated a plain `np.zeros` array
    instead, so the depot inventory had no merge semantics at all — in a paper whose
    whole framing is decentralised coordination under partition.
  * **`mmc_metrics` was dead code.** The runner re-derived Little's Law by hand from
    raw accumulators, which is both a second implementation and the one that does
    *not* raise on a degenerate service rate (`mmc_metrics` does).
  * It could not scale. The micro→depot assignment was an
    `argmin` over a **Python list comprehension of `np.hypot`** — O(micro × depots)
    in the interpreter, every step.

So the orchestration moves here, next to the decisions it drives. The runner keeps
the *scheduler choice* (it validates `hma` / `fcfs` / `milp_ortools` up front, and
`milp_ortools` needs an OR-Tools import it already guards) and injects it as a
callable, so this module stays free of optional heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from gossamer.algorithms.coordination.hma import DepotInventory, HMAParams, mmc_metrics

__all__ = ["HMAWorldConfig", "HMAWorld", "Assigner"]

#: (hauler_xy, hauler_soc, depot_xy, depot_mass, params, capacity) -> per-hauler
#: depot index, or None for "unassigned". Supplied by the caller so the scheduler's
#: optional dependencies (OR-Tools) stay out of this module.
Assigner = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, HMAParams, float],
    Sequence[Optional[int]],
]


@dataclass
class HMAWorldConfig:
    depots_xy: List[Tuple[float, float]]
    printers_xy: List[Tuple[float, float]]
    role_micro: List[int]
    role_hauler: List[int]
    role_printer: List[int]
    hauler_capacity: float = 500.0
    hauler_soc: float = 0.9
    s_crit: float = 0.30
    #: Proximity at which a micro deposits / a hauler loads or unloads.
    interaction_radius: float = 20.0
    #: kg a micro agent deposits per depot visit.
    deposit_kg: float = 1.0
    #: Cap on the steering acceleration magnitude (the runner's `vec_towards` max_a).
    max_accel: float = 1.0
    #: Where a LOADED hauler steers.
    #:
    #: ``False`` (default) reproduces the inlined runner loop exactly: the target is
    #: redirected to a printer only on the step the hauler actually loads, and on
    #: every other step a loaded hauler steers back at its assigned DEPOT. A hauler
    #: therefore oscillates between depot and printer and delivers only when it
    #: happens to pass within ``interaction_radius`` of one. In a 2-hauler /
    #: 1-printer world it delivers NOTHING (see tests); the HMA presets deliver only
    #: because they scatter 90 printers, so a wandering hauler eventually strays into
    #: range of one BY ACCIDENT.
    #:
    #: ``True`` makes a loaded hauler steer at its printer, which is what the market
    #: is supposed to model.
    #:
    #: The default is the buggy behaviour ON PURPOSE. The HMA portfolio numbers
    #: (batches ``hma_paper`` / ``hma_load``) were produced by it, so flipping this is
    #: a RESULT change that must be re-run and reported — not smuggled in under a
    #: decomposition. See CLAUDE.md §2.2.
    deliver_to_printer: bool = False


class HMAWorld:
    """One generation's market state: depots, haulers, printers, and the queue.

    `step()` returns the acceleration for every agent and advances the market;
    `metrics()` reduces the accumulated queue state through `mmc_metrics`.
    """

    def __init__(self, cfg: HMAWorldConfig, num_agents: int, assigner: Assigner) -> None:
        if num_agents <= 0:
            raise ValueError("num_agents must be positive")
        self.cfg = cfg
        self.n = num_agents
        self._assign = assigner
        self._params = HMAParams(s_crit=cfg.s_crit)

        n_depots = len(cfg.depots_xy)
        # The CRDT the module has always shipped and nobody used. `available()` is
        # kg-quantised (it wraps a PNCounter), which is why the float buffer below
        # tracks the continuous mass while the CRDT tracks the reconcilable ledger.
        self.depots: List[DepotInventory] = [DepotInventory() for _ in range(n_depots)]
        self._inv = np.zeros(n_depots, dtype=float)

        self._depot_xy = (np.asarray(cfg.depots_xy, dtype=float)
                          if n_depots else np.zeros((0, 2)))
        self._printer_xy = (np.asarray(cfg.printers_xy, dtype=float)
                            if cfg.printers_xy else np.zeros((0, 2)))

        self.hauler_load: Dict[int, float] = {i: 0.0 for i in cfg.role_hauler}
        self._target: Dict[int, int] = {}
        self._last_change: Dict[int, int] = {}
        self.reassign_intervals: List[int] = []

        self.total_printed = 0.0
        self.arrivals_kg = 0.0
        self.departures_kg = 0.0
        self._queue_sum = 0.0
        self._queue_n = 0
        self.queue_max = 0.0

    # -- helpers ------------------------------------------------------------

    def _toward(self, pos_i: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
        """Unit-magnitude steering, capped by the remaining distance.

        Byte-for-byte the runner's `vec_towards`: `(v/|v|) * min(max_accel, |v|)`,
        with the `+1e-9` in the denominator (not a zero-guard branch), because the
        two differ in the last ULP and this is a refactor, not a renumbering.
        """
        tgt = np.array([target_xy[0], target_xy[1], 0.0])
        v = tgt - pos_i
        n = float(np.linalg.norm(v)) + 1e-9
        return (v / n) * min(self.cfg.max_accel, n)

    # -- the loop -----------------------------------------------------------

    def step(self, step_index: int, pos: np.ndarray) -> np.ndarray:
        accel = np.zeros_like(pos)
        r = self.cfg.interaction_radius
        n_depots = self._depot_xy.shape[0]

        # --- micro agents: haul raw mass to the nearest depot ---------------
        # Vectorised. This was an argmin over a Python list comprehension of
        # np.hypot per micro agent per step — O(micro x depots) in the interpreter.
        if self.cfg.role_micro and n_depots:
            micro = np.asarray(self.cfg.role_micro, dtype=int)
            d2 = np.sum((pos[micro, None, :2] - self._depot_xy[None, :, :]) ** 2, axis=-1)
            nearest = np.argmin(d2, axis=1)
            for k, idx in enumerate(micro):
                j = int(nearest[k])
                accel[idx] = self._toward(pos[idx], self._depot_xy[j])
                tgt = np.array([self._depot_xy[j][0], self._depot_xy[j][1], 0.0])
                if float(np.linalg.norm(pos[idx] - tgt)) < r:
                    kg = self.cfg.deposit_kg
                    self._inv[j] += kg
                    # Mirror into the CRDT: this is the replica-mergeable ledger.
                    self.depots[j] = self.depots[j].deposit(kg, replica=f"micro-{idx}")
                    self.arrivals_kg += kg  # Little's Law lambda

        # --- haulers: assigned depot -> load -> printer ----------------------
        haulers = sorted(self.cfg.role_hauler)
        if haulers and n_depots:
            ha_pos = pos[np.asarray(haulers, dtype=int), :2]
            soc = np.full(len(haulers), float(self.cfg.hauler_soc))
            # The auction bids on mass; a depot with nothing in it still has to be a
            # legal target, or an empty market assigns nobody anywhere.
            depot_mass = np.maximum(self._inv, 1.0)
            assigns = self._assign(ha_pos, soc, self._depot_xy, depot_mass,
                                   self._params, self.cfg.hauler_capacity)

            for local_i, idx in enumerate(haulers):
                j = assigns[local_i]
                if j is None:
                    j = int(np.argmax(self._inv))
                j = int(j)

                # reassignment-churn accounting
                prev = self._target.get(idx)
                if prev is None or prev != j:
                    if prev is not None:
                        self.reassign_intervals.append(step_index - self._last_change[idx])
                    self._last_change[idx] = step_index
                    self._target[idx] = j

                target_xy = self._depot_xy[j]
                depot_pos = np.array([target_xy[0], target_xy[1], 0.0])

                # Load at the depot, up to capacity, then aim at a printer.
                #
                # NOTE — the steering here is deliberately NOT "a loaded hauler heads
                # for a printer". The target is only redirected to the printer on the
                # step the hauler actually loads; on any other step a loaded hauler
                # still steers at its assigned DEPOT. That is the behaviour of the
                # inlined loop this replaces, and the HMA portfolio numbers were
                # produced by it, so reproducing it is what makes this a refactor
                # rather than a silent renumbering. It looks like a bug (see
                # CLAUDE.md §2.2) and it is tracked as one — but fixing it is a
                # RESULT change and must be run and reported as such, not smuggled in
                # under a decomposition.
                if float(np.linalg.norm(pos[idx] - depot_pos)) < r and self._inv[j] > 0.0:
                    if self._printer_xy.shape[0]:
                        target_xy = self._printer_xy[idx % self._printer_xy.shape[0]]
                    headroom = max(0.0, self.cfg.hauler_capacity - self.hauler_load[idx])
                    moved = min(headroom, float(self._inv[j]))
                    if moved > 0.0:
                        self._inv[j] -= moved
                        self.depots[j] = self.depots[j].withdraw(moved, replica=f"hauler-{idx}")
                        self.departures_kg += moved  # service
                        self.hauler_load[idx] += moved

                # Deliver if we are at a printer with a load.
                if self._printer_xy.shape[0] and self.hauler_load[idx] > 0.0:
                    pxy = self._printer_xy[idx % self._printer_xy.shape[0]]
                    printer_pos = np.array([pxy[0], pxy[1], 0.0])
                    if float(np.linalg.norm(pos[idx] - printer_pos)) < r:
                        self.total_printed += self.hauler_load[idx]
                        self.hauler_load[idx] = 0.0
                    elif self.cfg.deliver_to_printer:
                        # The fix: a loaded hauler actually goes to its printer.
                        target_xy = pxy

                accel[idx] = self._toward(pos[idx], target_xy)

        # --- depot buffer occupancy (Little's Law L) -------------------------
        if self._inv.size:
            self._queue_sum += float(self._inv.mean())
            self._queue_n += 1
            self.queue_max = max(self.queue_max, float(self._inv.max()))

        return accel

    # -- reduction ----------------------------------------------------------

    @property
    def depot_inventory(self) -> np.ndarray:
        return self._inv

    def metrics(self, sim_seconds: float, movement_energy_j: float) -> Dict[str, float]:
        """Reduce the market state.

        Reports the queue TWO ways, because they are two different quantities and the
        paper's decoupling argument uses both:

          * ``hma_depot_wait_time_s_mean`` — the MEASURED wait, Little's Law applied
            to the observed buffer: ``W = L / lambda``. This is what the runner
            computed and what the HMA portfolio numbers report; it is unchanged here.
          * ``hma_depot_mmc_*`` — the THEORETICAL M/M/c steady state from the module's
            own `mmc_metrics`, which the runner never called despite the model being
            the paper's framing. New, additive.

        Where they disagree is itself the finding: `mmc_metrics` reports rho >= 1 (an
        unstable queue whose wait diverges), while the measured L/lambda always
        returns a finite, comfortable-looking number.
        """
        sim_seconds = max(1e-9, float(sim_seconds))
        n_haulers = len(self.cfg.role_hauler)

        queue_mean = (self._queue_sum / self._queue_n) if self._queue_n else 0.0
        lam = self.arrivals_kg / sim_seconds          # kg/s into the buffer
        mu_total = self.departures_kg / sim_seconds   # kg/s out of it

        out = {
            "hma_total_printed_kg": float(self.total_printed),
            "hma_energy_per_kg": (float(movement_energy_j / self.total_printed)
                                  if self.total_printed > 0 else 0.0),
            "hma_reallocation_latency_steps_mean": (
                float(np.mean(self.reassign_intervals)) if self.reassign_intervals else 0.0),
            "hma_depot_queue_depth_mean_kg": float(queue_mean),
            "hma_depot_queue_depth_max_kg": float(self.queue_max),
            "hma_depot_arrival_rate_kg_per_s": float(lam),
            "hma_depot_service_rate_kg_per_s": float(mu_total),
            "hma_num_servers_haulers": int(n_haulers),
        }

        # The MEASURED wait: Little's Law on the observed buffer. Byte-for-byte the
        # runner's reduction, so the published HMA numbers are reproduced exactly.
        out["hma_depot_wait_time_s_mean"] = float(queue_mean / lam) if lam > 0 else 0.0

        # The THEORETICAL M/M/c steady state. Additive — the runner never computed it.
        # Per-hauler service rate is what M/M/c wants; the accumulator is aggregate.
        mu_per_server = (mu_total / n_haulers) if n_haulers else 0.0
        if mu_per_server <= 0.0 or n_haulers <= 0:
            out["hma_depot_mmc_utilization"] = None
            out["hma_depot_mmc_saturated"] = None
            out["hma_depot_mmc_wait_time_s"] = None
            return out

        mmc = mmc_metrics(lam, mu_per_server, n_haulers)
        rho = float(mmc["rho"])
        out["hma_depot_mmc_utilization"] = rho
        # rho >= 1 means arrivals outrun service: the buffer grows without bound and
        # the steady-state wait diverges. mmc_metrics says so (inf) where the measured
        # L/lambda above still returns a finite, comfortable number for the very same
        # queue. Surface saturation as a flag rather than shipping a non-JSON
        # `Infinity` into experiment.json.
        out["hma_depot_mmc_saturated"] = bool(rho >= 1.0)
        out["hma_depot_mmc_wait_time_s"] = (
            None if not np.isfinite(mmc["W"]) else float(mmc["W"]))
        return out
