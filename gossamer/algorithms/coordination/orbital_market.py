"""The orbital workload market — HMA retargeted to compute/power/thermal/downlink.

This is roadmap 4.1: the terrestrial market's *mechanism* (the canonical
energy-aware bid utility, greedy batch clearing, and a CP-SAT central
comparator) applied to the resource problem an orbital compute constellation
actually has. The mapping from the lunar-construction market:

    depot mass          ->  queued compute jobs (the work to allocate)
    hauler capacity     ->  per-satellite compute slots
    hauler SOC          ->  battery state of charge (now *dynamic*: solar in,
                            jobs out, eclipse-driven)
    (new)               ->  thermal headroom (a hot satellite is suppressed the
                            same way a flat battery is)
    printer delivery    ->  downlink through ground-contact windows

The canonical utility keeps its published form (HMA paper §3.2):

    U_ij = alpha * (value_j / energy_j) * sigma_soc(SOC_i) * sigma_T(T_i)
           - beta * wait_j

`sigma_soc` is the same :func:`gossamer.algorithms.coordination.hma.soc_sigmoid`;
`sigma_T` is the identical sigmoid on thermal headroom. In this version every
job's value is its compute energy, so value/energy == 1 and satellites are
differentiated purely by their *state* — which is the honest formulation for a
homogeneous formation: the market's job is to route work toward satellites that
can currently afford it. Heterogeneous job values slot into `value_j` later
without touching the mechanism.

⚠️ SCOPING — read before citing any number from this module (DOCS §2.2, §3.5,
standing guardrails #1 and #6):

* **The market is regime-conditional, and one of its regimes is a loss.** The
  20-seed terrestrial measurement (batch ``hma_remeasure``) found the auction
  beats FCFS ONLY under scarcity (+33.6% at cap 10), is a null at the published
  operating point, and is significantly WORSE (-11.2%) when over-provisioned.
  This module therefore *measures* its own regime — `metrics()` reports the
  offered-load / net-supply energy ratio and a regime label — so every result
  carries its precondition as data. An orbital compute constellation is power-
  and thermal-scarce by construction, which is why the scarce regime is the one
  N1 lives in; but if your workload comes out `abundant`, expect the market to
  LOSE to FCFS, because that is what it measurably does.
* **No delay claim.** A 1-km formation is fully connected (< 0.1 ms effective
  delay; the cliff is at 10-20 s). This market assumes every bid is heard the
  step it is made. The delay science belongs to N3 and to the DCC papers.
* **Not device fidelity.** Power is `panel_w * illumination`, the battery is an
  energy integrator, the thermal model is lumped first-order with compute
  dissipation only (solar/albedo loads are folded into the sink temperature),
  and downlink is a rate through a boolean visibility window. These are
  algorithmic-research budgets, honest at the coordination-claim tier, and no
  claim above that tier survives them.
* **Architecture:** this module holds the *algorithm*. The illumination and
  ground-visibility series it consumes are geometry, and geometry belongs to
  Orrery (`orrery.astro.power`) — this module never imports it. Anything
  driving this world from real orbits does so by handing arrays across the
  seam, exactly like `ContactPlan.to_gossamer()`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from gossamer.algorithms.coordination.hma import soc_sigmoid

__all__ = [
    "Job",
    "MarketView",
    "OrbitalMarketParams",
    "OrbitalMarketWorld",
    "OrbitalWorldConfig",
    "SatelliteConfig",
    "Scheduler",
    "auction_scheduler",
    "cpsat_scheduler",
    "fcfs_scheduler",
    "greedy_soc_scheduler",
    "poisson_workload",
    "regime_label",
    "scarcity_ratio",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class OrbitalMarketParams:
    """Knobs of the canonical utility, orbital instantiation.

    ``alpha``/``beta``/``k_soc``/``s_crit`` keep their meanings (and defaults)
    from :class:`gossamer.algorithms.coordination.hma.HMAParams`. The thermal
    pair mirrors the SOC pair: ``t_crit_c`` is the temperature at which a
    satellite's bid is half-suppressed, ``k_thermal`` how sharply.
    """
    alpha: float = 1.0
    beta: float = 0.05          # weight on queue wait (s^-1)
    k_soc: float = 12.0
    s_crit: float = 0.30
    k_thermal: float = 0.5      # per degC
    t_crit_c: float = 60.0


@dataclass(frozen=True)
class Job:
    """One compute job. ``duration_s`` of runtime at ``power_w``, then
    ``data_out_mb`` must be downlinked before the job counts as delivered."""
    job_id: int
    arrival_s: float
    duration_s: float
    power_w: float
    data_out_mb: float

    @property
    def compute_energy_j(self) -> float:
        return self.duration_s * self.power_w


@dataclass(frozen=True)
class SatelliteConfig:
    """One satellite's budgets. Homogeneous across the formation in v1."""
    panel_w: float = 400.0
    base_load_w: float = 50.0
    battery_j: float = 2.16e6          # 600 Wh
    soc0: float = 0.9
    slots: int = 2                     # concurrent jobs
    heat_capacity_j_per_k: float = 8.0e4
    radiator_w_per_k: float = 15.0
    t_sink_c: float = 20.0             # lumped environment (incl. solar load)
    t0_c: float = 20.0
    t_max_c: float = 70.0              # hard throttle above this
    downlink_mbps: float = 100.0       # only while ground-visible
    dissipation_frac: float = 1.0      # compute power -> heat


@dataclass(frozen=True)
class OrbitalWorldConfig:
    n_sats: int
    dt: float = 1.0
    sat: SatelliteConfig = field(default_factory=SatelliteConfig)
    params: OrbitalMarketParams = field(default_factory=OrbitalMarketParams)


# --------------------------------------------------------------------------
# The scheduler seam
# --------------------------------------------------------------------------

class MarketView(NamedTuple):
    """What a scheduler is allowed to see: per-satellite state plus the queue.

    Deliberately global — the formation is fully connected (see the module
    header), so a decentralized auction and a central planner legitimately see
    the same state. What separates them is the *mechanism*, not the
    information. Partial views belong to the delay-coupled runner and N3.
    """
    now_s: float
    soc: np.ndarray            # (N,) in [0, 1]
    temp_c: np.ndarray         # (N,)
    free_slots: np.ndarray     # (N,) int
    illumination: np.ndarray   # (N,) in [0, 1]
    net_supply_w: np.ndarray   # (N,) panel*illum - base_load, can be negative
    queue: Tuple[Job, ...]     # pending jobs, arrival order
    params: OrbitalMarketParams


#: (view) -> [(queue_index, sat_index), ...]. The WORLD validates every pair
#: (free slot, job not already taken) and silently drops invalid ones, so a
#: buggy scheduler cannot oversubscribe a satellite — it can only waste its own
#: assignments.
Scheduler = Callable[[MarketView], Sequence[Tuple[int, int]]]


def _utilities(view: MarketView) -> np.ndarray:
    """The canonical utility for every (job, sat) pair; shape (len(queue), N).

    value/energy is 1 in v1 (value := compute energy), so the state sigmoids
    carry the satellite axis and the wait term carries the job axis. Kept as
    one function so the auction and the CP-SAT comparator provably optimize
    THE SAME objective — the head-to-head is mechanism vs mechanism, not
    objective vs objective.
    """
    p = view.params
    sig_soc = soc_sigmoid(view.soc, p.s_crit, p.k_soc)                    # (N,)
    # Same sigmoid, thermal axis: headroom = t_crit - T, half-suppressed at 0.
    sig_t = soc_sigmoid(view.temp_c * -1.0 + p.t_crit_c, 0.0, p.k_thermal)
    sat_term = p.alpha * sig_soc * sig_t                                  # (N,)
    wait = np.array([view.now_s - j.arrival_s for j in view.queue])       # (Q,)
    # Older jobs bid HIGHER (the wait term is a bonus to the pair, exactly as
    # the terrestrial market's -beta*(t_arrival+t_queue) is a malus): the
    # market must not starve old jobs to farm easy new ones.
    return sat_term[None, :] + p.beta * wait[:, None]


def _eligible(view: MarketView) -> np.ndarray:
    """The reserve price, orbital form: (Q, N) bool — may sat s bid on job q?

    The terrestrial auction refuses to clear a non-positive bid (`if u <= 0:
    break`), which is what stops it from behaving like FCFS with extra steps.
    The orbital utility is positive by construction (state sigmoids + a wait
    bonus), so the reserve moves into an explicit solvency rule: **a satellite
    may not bid on a job it cannot power today** — battery above ``s_crit`` OR
    live solar surplus covering the job's draw. Without this, the first test
    that put a flat, eclipsed satellite next to a sunlit one showed the auction
    parking jobs on the dark one exactly like FCFS: an always-positive bid is
    no market at all.

    FCFS and greedy-SOC deliberately do NOT apply this rule — being blind to it
    is what makes them the baselines.
    """
    solvent = view.soc > view.params.s_crit                                # (N,)
    power = np.array([j.power_w for j in view.queue])                      # (Q,)
    covered = view.net_supply_w[None, :] >= power[:, None]                 # (Q, N)
    return solvent[None, :] | covered


def auction_scheduler(view: MarketView) -> List[Tuple[int, int]]:
    """The decentralized market: greedy clearing by canonical utility.

    Every satellite with a free slot bids on every queued job; pairs clear
    highest-utility-first, one job per slot. Ties break by (queue index, sat
    index) so the outcome is deterministic. This is `energy_aware_auction`'s
    clearing rule with the orbital utility.
    """
    if not view.queue:
        return []
    U = _utilities(view)
    ok = _eligible(view)
    order = sorted(
        ((float(U[q, s]), q, s)
         for q in range(len(view.queue))
         for s in range(len(view.soc))
         if view.free_slots[s] > 0 and ok[q, s]),
        key=lambda t: (-t[0], t[1], t[2]),
    )
    slots = view.free_slots.copy()
    taken: set = set()
    out: List[Tuple[int, int]] = []
    for _u, q, s in order:
        if q in taken or slots[s] <= 0:
            continue
        taken.add(q)
        slots[s] -= 1
        out.append((q, s))
    return out


def fcfs_scheduler(view: MarketView) -> List[Tuple[int, int]]:
    """The naive baseline: oldest job to the lowest-index free slot.

    Deliberately ignores SOC, temperature and illumination — that blindness is
    the thing the market is measured against. Do not 'improve' it; a smarter
    FCFS is a different comparator.
    """
    slots = view.free_slots.copy()
    out: List[Tuple[int, int]] = []
    for q in range(len(view.queue)):
        for s in range(len(slots)):
            if slots[s] > 0:
                slots[s] -= 1
                out.append((q, s))
                break
    return out


def greedy_soc_scheduler(view: MarketView) -> List[Tuple[int, int]]:
    """One-signal greedy: oldest job to the highest-SOC free satellite.

    The middle rung between FCFS (no signals) and the auction (all signals):
    it sees the battery but not temperature, wait pressure, or the utility's
    energy normalisation.
    """
    slots = view.free_slots.copy()
    out: List[Tuple[int, int]] = []
    for q in range(len(view.queue)):
        best, best_soc = -1, -1.0
        for s in range(len(slots)):
            if slots[s] > 0 and view.soc[s] > best_soc:
                best, best_soc = s, float(view.soc[s])
        if best < 0:
            break
        slots[best] -= 1
        out.append((q, best))
    return out


def cpsat_scheduler(view: MarketView, *, time_budget_s: float = 1.0,
                    utility_scale: int = 1000) -> List[Tuple[int, int]]:
    """The central comparator: CP-SAT maximizing the SAME canonical utility.

    Answers the reviewer's question — how far is the decentralized clearing
    from the centralized optimum of its own objective — mirroring
    :func:`gossamer.algorithms.coordination.milp_baseline.milp_assignment`.
    OR-Tools is imported lazily; utilities are integer-scaled (CP-SAT is
    integer-only), so optimality is up to the ``utility_scale`` quantization.

    Determinism caveat (same reason `milp_ortools` is excluded from the HMA
    fingerprint): under a wall-clock budget the incumbent solution can depend
    on machine speed. Results are valid; bit-reproducibility is not promised.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError(
            "cpsat_scheduler requires OR-Tools ('pip install ortools', or the "
            "gossamer[milp] extra)."
        ) from e
    if not view.queue:
        return []

    U = _utilities(view)
    ok = _eligible(view)  # the central planner honors the same solvency rule
    Q, N = U.shape
    model = cp_model.CpModel()
    x = {}
    for q in range(Q):
        for s in range(N):
            if view.free_slots[s] > 0 and ok[q, s]:
                x[q, s] = model.NewBoolVar(f"x_{q}_{s}")
    for q in range(Q):
        row = [x[q, s] for s in range(N) if (q, s) in x]
        if row:
            model.Add(sum(row) <= 1)
    for s in range(N):
        col = [x[q, s] for q in range(Q) if (q, s) in x]
        if col:
            model.Add(sum(col) <= int(view.free_slots[s]))
    model.Maximize(sum(int(round(float(U[q, s]) * utility_scale)) * var
                       for (q, s), var in x.items()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_budget_s)
    solver.parameters.num_search_workers = 1  # keep the search single-threaded
    solver.Solve(model)
    return sorted((q, s) for (q, s), var in x.items() if solver.Value(var) == 1)


# --------------------------------------------------------------------------
# Workload + regime
# --------------------------------------------------------------------------

def poisson_workload(rng: np.random.Generator, *, rate_per_s: float,
                     horizon_s: float, duration_s: float = 120.0,
                     power_w: float = 150.0, data_out_mb: float = 50.0) -> List[Job]:
    """Poisson job arrivals over the horizon. Seeded via ``rng``; the world
    itself draws no randomness, so the workload is the run's only entropy."""
    jobs: List[Job] = []
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / rate_per_s))
        if t >= horizon_s:
            break
        jobs.append(Job(job_id=len(jobs), arrival_s=t, duration_s=duration_s,
                        power_w=power_w, data_out_mb=data_out_mb))
    return jobs


def scarcity_ratio(jobs: Sequence[Job], illumination: np.ndarray,
                   sat: SatelliteConfig, dt: float) -> float:
    """Offered compute energy over net available energy — THE regime number.

    numerator: sum of every arriving job's compute energy.
    denominator: sum over (step, sat) of max(0, panel*illum - base_load)*dt —
    the energy actually available for compute after keeping the lights on.

    rho >= 1 is genuine scarcity: the offered load cannot be served even by a
    perfect scheduler running every panel flat out. This is the orbital
    analogue of the terrestrial capacity axis, and it is the number every N1
    claim must be conditioned on (standing guardrail #6).
    """
    demand = float(sum(j.compute_energy_j for j in jobs))
    net_w = np.maximum(0.0, sat.panel_w * np.asarray(illumination, dtype=float)
                       - sat.base_load_w)
    supply = float(net_w.sum() * dt)
    return demand / supply if supply > 0 else float("inf")


def regime_label(rho: float) -> str:
    """Band definitions (definitions, not measurements): the terrestrial
    result says expect the market to WIN when ``scarce``, tie-or-null when
    ``contended``, and LOSE when ``abundant``."""
    if rho >= 1.0:
        return "scarce"
    if rho >= 0.6:
        return "contended"
    return "abundant"


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------

@dataclass
class _Running:
    job: Job
    remaining_s: float
    start_s: float
    eclipse_progress_s: float = 0.0   # runtime accrued while illum < 0.1


@dataclass
class _Done:
    job: Job
    sat: int
    start_s: float
    finish_s: float
    eclipse_progress_s: float
    remaining_mb: float               # downlink backlog for this job
    delivered_s: Optional[float] = None


class OrbitalMarketWorld:
    """Discrete-time market: arrivals -> schedule -> power -> thermal -> downlink.

    Deterministic by construction: the world draws no randomness (the workload
    carries the seed), schedulers are pure functions of the view, and every
    per-step update walks satellites and jobs in fixed index order.

    ``illumination`` and ``ground_visible`` are (T, N) arrays — geometry
    computed elsewhere (Orrery for real orbits, literals in tests). ``T`` must
    cover ``horizon_steps``.
    """

    def __init__(self, cfg: OrbitalWorldConfig, jobs: Sequence[Job],
                 scheduler: Scheduler, illumination: np.ndarray,
                 ground_visible: np.ndarray) -> None:
        self.cfg = cfg
        self._sched = scheduler
        self._jobs = sorted(jobs, key=lambda j: (j.arrival_s, j.job_id))
        self._illum = np.asarray(illumination, dtype=float)
        self._gvis = np.asarray(ground_visible, dtype=bool)
        n = cfg.n_sats
        if self._illum.ndim != 2 or self._illum.shape[1] != n:
            raise ValueError(f"illumination must be (T, {n}); got {self._illum.shape}")
        if self._gvis.shape != self._illum.shape:
            raise ValueError("ground_visible must match illumination's shape")

        self.soc = np.full(n, float(cfg.sat.soc0))
        self.temp_c = np.full(n, float(cfg.sat.t0_c))
        self.running: List[List[_Running]] = [[] for _ in range(n)]
        self.queue: List[Job] = []
        self.done: List[_Done] = []
        self._downlink: List[List[_Done]] = [[] for _ in range(n)]  # FIFO per sat

        self._step_i = 0
        self._next_arrival = 0
        # accounting
        self.energy_drawn_j = 0.0        # base + compute actually consumed
        self.thermal_violation_sat_steps = 0
        self.brownout_sat_steps = 0
        self.starved_job_steps = 0
        self._wait_sum_s = 0.0           # arrival -> start, over started jobs
        self._wait_n = 0

    # -- the step ------------------------------------------------------------

    def step(self) -> None:
        cfg, sat = self.cfg, self.cfg.sat
        i_step = self._step_i
        if i_step >= self._illum.shape[0]:
            raise IndexError("stepped past the illumination series' horizon")
        now = i_step * cfg.dt
        illum = self._illum[i_step]
        gvis = self._gvis[i_step]
        dt = cfg.dt

        # 1. Arrivals whose time has come join the queue (stable order).
        while (self._next_arrival < len(self._jobs)
               and self._jobs[self._next_arrival].arrival_s <= now):
            self.queue.append(self._jobs[self._next_arrival])
            self._next_arrival += 1

        # 2. Scheduling. The world enforces legality; the scheduler only ranks.
        if self.queue:
            free = np.array([sat.slots - len(r) for r in self.running], dtype=int)
            view = MarketView(now_s=now, soc=self.soc.copy(),
                              temp_c=self.temp_c.copy(), free_slots=free.copy(),
                              illumination=illum.copy(),
                              net_supply_w=sat.panel_w * illum - sat.base_load_w,
                              queue=tuple(self.queue), params=cfg.params)
            taken: set = set()
            for q, s in self._sched(view):
                if (q in taken or not (0 <= q < len(self.queue))
                        or not (0 <= s < cfg.n_sats) or free[s] <= 0):
                    continue
                taken.add(q)
                free[s] -= 1
                job = self.queue[q]
                self.running[s].append(_Running(job=job, remaining_s=job.duration_s,
                                                start_s=now))
                self._wait_sum_s += now - job.arrival_s
                self._wait_n += 1
            if taken:
                self.queue = [j for q, j in enumerate(self.queue) if q not in taken]

        # 3+4. Power and thermal, satellite by satellite in index order.
        for s in range(cfg.n_sats):
            supply_j = sat.panel_w * float(illum[s]) * dt
            bank_j = float(self.soc[s]) * sat.battery_j + supply_j

            # Base load first: below it the bus browns out and nothing runs.
            brownout = bank_j < sat.base_load_w * dt
            if brownout:
                self.brownout_sat_steps += 1
                bank_j = max(0.0, bank_j)  # whatever trickles in stays banked
            else:
                bank_j -= sat.base_load_w * dt
                self.energy_drawn_j += sat.base_load_w * dt

            # Thermal throttle is decided on the PRE-step temperature: a
            # satellite that ended the last step too hot does no work this one.
            throttled = self.temp_c[s] > sat.t_max_c
            if throttled:
                self.thermal_violation_sat_steps += 1

            heat_in_j = 0.0
            finished: List[int] = []
            for k, run in enumerate(self.running[s]):
                need_j = run.job.power_w * dt
                if brownout or throttled:
                    self.starved_job_steps += 1
                    continue
                if bank_j < need_j:
                    self.starved_job_steps += 1
                    continue
                bank_j -= need_j
                self.energy_drawn_j += need_j
                heat_in_j += need_j * sat.dissipation_frac
                run.remaining_s -= dt
                if float(illum[s]) < 0.1:
                    run.eclipse_progress_s += dt
                if run.remaining_s <= 1e-9:
                    finished.append(k)
            for k in reversed(finished):
                run = self.running[s].pop(k)
                self._downlink[s].append(_Done(
                    job=run.job, sat=s, start_s=run.start_s,
                    finish_s=now + dt, eclipse_progress_s=run.eclipse_progress_s,
                    remaining_mb=run.job.data_out_mb))

            # Battery banks what is left, sheds what it cannot hold.
            self.soc[s] = min(1.0, bank_j / sat.battery_j)

            # Lumped first-order node, explicit Euler on the pre-step T.
            radiated_j = sat.radiator_w_per_k * (self.temp_c[s] - sat.t_sink_c) * dt
            self.temp_c[s] += (heat_in_j - radiated_j) / sat.heat_capacity_j_per_k

        # 5. Downlink: FIFO per satellite while ground-visible.
        for s in range(cfg.n_sats):
            if not self._gvis[i_step, s] or not self._downlink[s]:
                continue
            budget_mb = sat.downlink_mbps * dt / 8.0
            while budget_mb > 1e-12 and self._downlink[s]:
                head = self._downlink[s][0]
                sent = min(budget_mb, head.remaining_mb)
                head.remaining_mb -= sent
                budget_mb -= sent
                if head.remaining_mb <= 1e-9:
                    head.delivered_s = now + dt
                    self.done.append(head)
                    self._downlink[s].pop(0)

        self._step_i += 1

    def run(self, horizon_steps: Optional[int] = None) -> Dict[str, object]:
        steps = self._illum.shape[0] if horizon_steps is None else int(horizon_steps)
        for _ in range(steps - self._step_i):
            self.step()
        return self.metrics()

    # -- reduction -------------------------------------------------------------

    def metrics(self) -> Dict[str, object]:
        """Reduce the run. Missing quantities are None, never a plausible zero
        (the house rule): a run that delivered nothing has no makespan, not a
        makespan of 0.0."""
        cfg = self.cfg
        delivered = [d for d in self.done if d.delivered_s is not None]
        completed_backlog = sum(len(q) for q in self._downlink)
        backlog_mb = float(sum(d.remaining_mb for q in self._downlink for d in q))
        n_arrived = self._next_arrival
        n_running = sum(len(r) for r in self.running)

        # The regime, measured from what this run actually offered and had —
        # over the steps actually run, not the series' full horizon.
        steps_run = self._step_i
        rho = scarcity_ratio(self._jobs[:n_arrived],
                             self._illum[:steps_run], cfg.sat, cfg.dt)

        first_arrival = self._jobs[0].arrival_s if self._jobs else None
        makespan = (max(d.delivered_s for d in delivered) - first_arrival
                    if delivered and first_arrival is not None else None)
        flow = ([d.delivered_s - d.job.arrival_s for d in delivered]
                if delivered else None)

        return {
            "market_scarcity_rho": float(rho),
            "market_regime": regime_label(rho),
            "jobs_arrived": int(n_arrived),
            "jobs_started": int(self._wait_n),
            "jobs_completed": int(len(self.done) + completed_backlog),
            "jobs_delivered": int(len(delivered)),
            "jobs_queued_final": int(len(self.queue)),
            "jobs_running_final": int(n_running),
            "makespan_s": None if makespan is None else float(makespan),
            "job_wait_s_mean": (self._wait_sum_s / self._wait_n
                                if self._wait_n else None),
            "job_flow_s_mean": None if flow is None else float(np.mean(flow)),
            "energy_drawn_j": float(self.energy_drawn_j),
            "energy_per_delivered_job_j": (self.energy_drawn_j / len(delivered)
                                           if delivered else None),
            "jobs_through_eclipse": int(sum(1 for d in delivered
                                            if d.eclipse_progress_s > 0.0)),
            "thermal_violation_sat_steps": int(self.thermal_violation_sat_steps),
            "brownout_sat_steps": int(self.brownout_sat_steps),
            "starved_job_steps": int(self.starved_job_steps),
            "downlink_backlog_mb": backlog_mb,
            "soc_min": float(self.soc.min()) if cfg.n_sats else None,
            "soc_mean_final": float(self.soc.mean()) if cfg.n_sats else None,
            "temp_c_max_final": float(self.temp_c.max()) if cfg.n_sats else None,
        }
