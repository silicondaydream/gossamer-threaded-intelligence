"""Tests for the orbital workload market (roadmap 4.1).

Mechanism tests, not science claims: where a scheduler comparison appears it is
a CONSTRUCTED deterministic case that checks the mechanism does what its
docstring says (energy-blind FCFS stalls on a dark satellite; the auction routes
around it) — the measured, seeded, multi-cell comparison belongs to a batch, and
its regime precondition is guardrail #6.
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.orbital_market import (
    Job,
    MarketView,
    OrbitalMarketParams,
    OrbitalMarketWorld,
    OrbitalWorldConfig,
    SatelliteConfig,
    MarketAdversary,
    NoCorruptionFiredError,
    auction_scheduler,
    fcfs_scheduler,
    greedy_soc_scheduler,
    lying_auction_scheduler,
    poisson_workload,
    regime_label,
    scarcity_ratio,
    verify_corruption_fired,
)


def _world(n_sats, jobs, scheduler, illum, gvis=None, sat=None, params=None):
    illum = np.asarray(illum, dtype=float)
    if gvis is None:
        gvis = np.ones_like(illum, dtype=bool)
    cfg = OrbitalWorldConfig(
        n_sats=n_sats, dt=1.0,
        sat=sat or SatelliteConfig(),
        params=params or OrbitalMarketParams(),
    )
    return OrbitalMarketWorld(cfg, jobs, scheduler, illum, np.asarray(gvis, dtype=bool))


def _steady(n_steps, n_sats, value=1.0):
    return np.full((n_steps, n_sats), float(value))


# --------------------------------------------------------------------------
# Power / battery
# --------------------------------------------------------------------------

def test_battery_drains_at_base_load_in_eclipse_and_clips_at_full_in_sun():
    sat = SatelliteConfig(panel_w=400.0, base_load_w=50.0, battery_j=1.0e5, soc0=0.5)
    dark = _world(1, [], fcfs_scheduler, _steady(10, 1, 0.0), sat=sat)
    dark.run()
    # 10 steps x 50 W out of a 1e5 J battery starting half-full.
    expected = 0.5 - 10 * 50.0 / 1.0e5
    assert dark.soc[0] == pytest.approx(expected)

    lit = _world(1, [], fcfs_scheduler, _steady(400, 1, 1.0), sat=sat)
    lit.run()
    assert lit.soc[0] == 1.0  # net +350 W charges to full, then sheds


def test_brownout_pauses_jobs_and_is_counted_never_negative_soc():
    sat = SatelliteConfig(panel_w=0.0, base_load_w=50.0, battery_j=200.0, soc0=1.0)
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=5.0, power_w=10.0, data_out_mb=1.0)]
    w = _world(1, jobs, fcfs_scheduler, _steady(20, 1, 0.0), sat=sat)
    m = w.run()
    # The bus browns out once the bank cannot cover base load; the residual
    # charge below that threshold stays banked (20 J of a 200 J battery).
    assert w.soc[0] == pytest.approx(0.1)
    assert m["brownout_sat_steps"] > 0
    assert m["starved_job_steps"] > 0
    assert m["jobs_delivered"] == 0


# --------------------------------------------------------------------------
# Eclipse: pause on empty battery, resume in sun, and count the crossing
# --------------------------------------------------------------------------

def test_job_survives_eclipse_on_battery_and_is_counted_through_eclipse():
    # Battery holds base+job for well under the eclipse; the job must stall
    # (starved), then finish once the sun comes back.
    sat = SatelliteConfig(panel_w=400.0, base_load_w=50.0, battery_j=2000.0,
                          soc0=1.0, slots=1)
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=30.0, power_w=100.0,
                data_out_mb=1.0)]
    illum = np.concatenate([np.zeros((20, 1)), np.ones((60, 1))])
    w = _world(1, jobs, fcfs_scheduler, illum, sat=sat)
    m = w.run()
    assert m["jobs_delivered"] == 1
    assert m["starved_job_steps"] > 0        # it stalled in the dark
    assert m["jobs_through_eclipse"] == 1    # and it accrued eclipse runtime


# --------------------------------------------------------------------------
# Thermal
# --------------------------------------------------------------------------

def test_hot_satellite_throttles_then_recovers():
    # Tiny thermal mass + weak radiator: the job cooks the sat past t_max,
    # progress halts (violation steps count), the radiator cools it back down,
    # and the job eventually completes.
    sat = SatelliteConfig(panel_w=400.0, base_load_w=10.0, battery_j=1.0e7,
                          soc0=1.0, slots=1, heat_capacity_j_per_k=100.0,
                          radiator_w_per_k=2.0, t_sink_c=20.0, t0_c=20.0,
                          t_max_c=40.0)
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=25.0, power_w=200.0,
                data_out_mb=1.0)]
    w = _world(1, jobs, fcfs_scheduler, _steady(400, 1, 1.0), sat=sat)
    m = w.run()
    assert m["thermal_violation_sat_steps"] > 0
    assert m["jobs_delivered"] == 1
    # Throttling means the wall-clock exceeds the nominal duration.
    assert m["makespan_s"] > 25.0


# --------------------------------------------------------------------------
# Downlink
# --------------------------------------------------------------------------

def test_delivery_is_gated_on_ground_visibility():
    sat = SatelliteConfig(downlink_mbps=80.0)  # 10 MB/s
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=3.0, power_w=10.0,
                data_out_mb=40.0)]
    # No window at all: completed, never delivered.
    dark = _world(1, jobs, fcfs_scheduler, _steady(30, 1, 1.0),
                  gvis=np.zeros((30, 1), dtype=bool), sat=sat)
    m = dark.run()
    assert m["jobs_completed"] == 1
    assert m["jobs_delivered"] == 0
    assert m["downlink_backlog_mb"] == pytest.approx(40.0)
    assert m["makespan_s"] is None            # no delivery -> no makespan, not 0.0

    # A window from step 20: 40 MB at 10 MB/s needs 4 visible steps.
    gvis = np.zeros((30, 1), dtype=bool)
    gvis[20:] = True
    lit = _world(1, jobs, fcfs_scheduler, _steady(30, 1, 1.0), gvis=gvis, sat=sat)
    m = lit.run()
    assert m["jobs_delivered"] == 1
    assert m["downlink_backlog_mb"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# The world enforces legality regardless of the scheduler
# --------------------------------------------------------------------------

def test_world_drops_illegal_assignments_from_a_hostile_scheduler():
    def hostile(view):
        # Same job twice, out-of-range sat, and more jobs than slots.
        return [(0, 0), (0, 1), (1, 99), (1, 0), (2, 0), (3, 0)]

    sat = SatelliteConfig(slots=2)
    jobs = [Job(job_id=i, arrival_s=0.0, duration_s=2.0, power_w=10.0,
                data_out_mb=1.0) for i in range(4)]
    w = _world(2, jobs, hostile, _steady(10, 2, 1.0), sat=sat)
    w.step()
    # job0 -> sat0, job1's sat99 dropped, job1 -> sat0 (second slot),
    # job2/job3 -> sat0 refused (full); nothing double-assigned.
    assert len(w.running[0]) == 2
    assert len(w.running[1]) == 0
    assert len(w.queue) == 2


# --------------------------------------------------------------------------
# Schedulers
# --------------------------------------------------------------------------

def _view(soc, temp, free, queue, now=100.0, net_supply_w=None):
    n = len(soc)
    if net_supply_w is None:
        net_supply_w = np.full(n, 350.0)   # sunlit default: plenty of surplus
    return MarketView(
        now_s=now, soc=np.asarray(soc, float), temp_c=np.asarray(temp, float),
        free_slots=np.asarray(free, int), illumination=np.ones(n),
        net_supply_w=np.asarray(net_supply_w, float),
        queue=tuple(queue), params=OrbitalMarketParams(),
    )


def _jobs(n, arrival=0.0):
    return [Job(job_id=i, arrival_s=arrival, duration_s=10.0, power_w=100.0,
                data_out_mb=1.0) for i in range(n)]


def test_fcfs_is_index_blind_and_greedy_follows_soc():
    v = _view(soc=[0.05, 0.95], temp=[25.0, 25.0], free=[1, 1], queue=_jobs(1))
    assert fcfs_scheduler(v) == [(0, 0)]           # index order, state-blind
    assert greedy_soc_scheduler(v) == [(0, 1)]     # follows the battery


def test_auction_avoids_flat_battery_and_hot_satellite():
    # Flat battery suppressed:
    v = _view(soc=[0.05, 0.95], temp=[25.0, 25.0], free=[1, 1], queue=_jobs(1))
    assert auction_scheduler(v) == [(0, 1)]
    # Hot satellite suppressed even at equal SOC:
    v = _view(soc=[0.9, 0.9], temp=[75.0, 25.0], free=[1, 1], queue=_jobs(1))
    assert auction_scheduler(v) == [(0, 1)]


def test_auction_refuses_insolvent_satellites_and_waits():
    # Every sat is below s_crit with no solar surplus: the market leaves the
    # job QUEUED rather than parking it where it will starve. (FCFS assigns
    # anyway — that blindness is the baseline's defining property.)
    v = _view(soc=[0.1, 0.2], temp=[25.0, 25.0], free=[1, 1], queue=_jobs(1),
              net_supply_w=[-50.0, -50.0])
    assert auction_scheduler(v) == []
    assert fcfs_scheduler(v) == [(0, 0)]
    # Solar surplus covering the job's draw restores eligibility even at low SOC.
    v = _view(soc=[0.1, 0.2], temp=[25.0, 25.0], free=[1, 1], queue=_jobs(1),
              net_supply_w=[-50.0, 200.0])
    assert auction_scheduler(v) == [(0, 1)]


def test_auction_clears_older_jobs_first_when_slots_are_scarce():
    old = Job(job_id=0, arrival_s=0.0, duration_s=10.0, power_w=100.0, data_out_mb=1.0)
    new = Job(job_id=1, arrival_s=99.0, duration_s=10.0, power_w=100.0, data_out_mb=1.0)
    v = _view(soc=[0.9], temp=[25.0], free=[1], queue=[new, old], now=100.0)
    assert auction_scheduler(v) == [(1, 0)]        # the older job wins the slot


def test_cpsat_matches_or_beats_the_auction_on_its_own_objective():
    pytest.importorskip("ortools")
    from gossamer.algorithms.coordination.orbital_market import (
        _utilities, cpsat_scheduler)

    v = _view(soc=[0.2, 0.6, 0.95], temp=[65.0, 40.0, 25.0], free=[1, 1, 1],
              queue=_jobs(3))
    U = _utilities(v)

    def total(pairs):
        return sum(float(U[q, s]) for q, s in pairs)

    a = auction_scheduler(v)
    c = cpsat_scheduler(v, time_budget_s=10.0)
    # Legality:
    assert len({q for q, _ in c}) == len(c)
    assert len({s for _, s in c}) == len(c)        # one slot each here
    # The central planner optimizes the SAME objective, so it cannot do worse
    # (up to the integer quantization of the utilities).
    assert total(c) >= total(a) - 1e-3


# --------------------------------------------------------------------------
# The constructed mechanism comparison (not a science claim)
# --------------------------------------------------------------------------

def test_auction_routes_around_a_dark_satellite_where_fcfs_stalls():
    # Sat 0: eclipsed, near-flat battery. Sat 1: sunlit, full. FCFS is
    # index-blind so it parks every job on sat 0, where they starve; the
    # auction's SOC sigmoid routes them to sat 1.
    sat = SatelliteConfig(panel_w=400.0, base_load_w=20.0, battery_j=5000.0,
                          slots=1, soc0=1.0)
    horizon = 60
    illum = np.zeros((horizon, 2))
    illum[:, 1] = 1.0
    jobs = [Job(job_id=i, arrival_s=0.0, duration_s=10.0, power_w=150.0,
                data_out_mb=1.0) for i in range(3)]

    def run(sched):
        w = _world(2, jobs, sched, illum, sat=sat)
        # sat 0 starts nearly flat; construct that by pre-draining.
        w.soc[0] = 0.05
        return w.run()

    m_fcfs = run(fcfs_scheduler)
    m_auct = run(auction_scheduler)
    assert m_auct["jobs_delivered"] > m_fcfs["jobs_delivered"]
    assert m_auct["makespan_s"] is not None


# --------------------------------------------------------------------------
# Regime measurement (guardrail #6 as code)
# --------------------------------------------------------------------------

def test_scarcity_ratio_is_demand_over_net_supply():
    sat = SatelliteConfig(panel_w=100.0, base_load_w=40.0)
    illum = np.ones((100, 2))                      # net 60 W x 2 sats x 100 s
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=60.0, power_w=100.0,
                data_out_mb=1.0)]                  # 6000 J offered
    assert scarcity_ratio(jobs, illum, sat, dt=1.0) == pytest.approx(0.5)
    # Eclipse halves the supply -> the SAME workload becomes scarce.
    illum[50:] = 0.0
    assert scarcity_ratio(jobs, illum, sat, dt=1.0) == pytest.approx(1.0)


def test_regime_label_bands():
    assert regime_label(1.2) == "scarce"
    assert regime_label(1.0) == "scarce"
    assert regime_label(0.8) == "contended"
    assert regime_label(0.3) == "abundant"


def test_metrics_report_the_measured_regime():
    sat = SatelliteConfig(panel_w=100.0, base_load_w=90.0, battery_j=1e6, soc0=1.0)
    jobs = [Job(job_id=0, arrival_s=0.0, duration_s=50.0, power_w=100.0,
                data_out_mb=1.0)]
    w = _world(1, jobs, auction_scheduler, _steady(100, 1, 1.0), sat=sat)
    m = w.run()
    # demand 5000 J vs net supply (100-90) x 100 = 1000 J -> deeply scarce.
    assert m["market_scarcity_rho"] == pytest.approx(5.0)
    assert m["market_regime"] == "scarce"


# --------------------------------------------------------------------------
# Determinism + workload
# --------------------------------------------------------------------------

def test_seeded_run_is_exactly_reproducible():
    def run():
        rng = np.random.default_rng(7)
        jobs = poisson_workload(rng, rate_per_s=0.05, horizon_s=200.0,
                                duration_s=30.0, power_w=120.0, data_out_mb=10.0)
        illum = np.tile(np.concatenate([np.ones(120), np.zeros(80)])[:, None], (1, 4))
        gvis = np.zeros((200, 4), dtype=bool)
        gvis[150:] = True
        w = _world(4, jobs, auction_scheduler, illum, gvis=gvis)
        return w.run()

    a, b = run(), run()
    assert a == b


def test_poisson_workload_is_seeded_and_bounded():
    rng = np.random.default_rng(3)
    jobs = poisson_workload(rng, rate_per_s=0.1, horizon_s=500.0)
    assert jobs, "expected at least one arrival at rate 0.1/s over 500 s"
    assert all(0.0 < j.arrival_s < 500.0 for j in jobs)
    assert [j.job_id for j in jobs] == list(range(len(jobs)))
    jobs2 = poisson_workload(np.random.default_rng(3), rate_per_s=0.1, horizon_s=500.0)
    assert jobs == jobs2


# --------------------------------------------------------------------------
# The corruption seam (N5). These are the locks, and the FIRST one is the
# important one: N1's 240-cell campaign is a published pin, so the seam has to
# be provably inert when off. Mirrors Leviathan's
# `SeuIsOffByDefaultAndConsumesNoRandomness`.
# --------------------------------------------------------------------------

def _adv_world(scheduler, adversary=None, n_sats=4, steps=40):
    """A deliberately ENERGY-TIGHT world: eclipse, and SOC just under s_crit.

    The scarcity is the point. The solvency rule only binds where a satellite
    cannot power the job, so in an abundant world a satellite that overestimates
    its own battery still completes the work and belief corruption leaves no
    trace -- the hook would look like a no-op while firing perfectly. Full
    eclipse (no supply) plus soc0 below the s_crit=0.30 reserve is the regime
    where a wrong belief actually buys a job the satellite cannot run.
    """
    jobs = [Job(job_id=i, arrival_s=float(i), duration_s=8.0, power_w=140.0,
                data_out_mb=1.0)
            for i in range(12)]
    illum = _steady(steps, n_sats, 0.0)          # eclipse: nothing coming in
    cfg = OrbitalWorldConfig(
        n_sats=n_sats, dt=1.0,
        sat=SatelliteConfig(soc0=0.25, battery_j=5.0e3),
        params=OrbitalMarketParams())
    w = OrbitalMarketWorld(cfg, jobs, scheduler, illum,
                           np.ones_like(illum, dtype=bool), adversary=adversary)
    for _ in range(steps):
        w.step()
    return w


def test_a_disabled_adversary_is_the_no_adversary_path_exactly():
    """The pin: off must be BIT-identical, not merely similar."""
    keys = ("jobs_delivered", "energy_drawn_j", "brownout_sat_steps",
            "starved_job_steps", "soc_min", "job_wait_s_mean",
            "thermal_violation_sat_steps")
    none_m = _adv_world(auction_scheduler, None).metrics()
    for disabled in (MarketAdversary(),
                     # enabled=False for every shape of "configured but inert"
                     MarketAdversary(faulty_fraction=0.5),
                     MarketAdversary(view_soc_error=0.4),
                     MarketAdversary(faulty_fraction=0.5, bid_inflation=1.0)):
        assert not disabled.enabled
        m = _adv_world(auction_scheduler, disabled).metrics()
        for k in keys:
            assert m[k] == none_m[k], f"{disabled} moved {k}"


def test_a_disabled_adversary_draws_no_randomness():
    """`marked` must not touch a generator when off.

    A generator constructed-but-unused is harmless today and a renumbering bug
    the moment anything else shares the stream, which is exactly how the
    published-number traps in this repo have started.
    """
    assert MarketAdversary().marked(81).size == 0
    assert MarketAdversary(faulty_fraction=0.5).marked(81).size == 0


def test_view_corruption_fires_and_reports_its_evidence():
    adv = MarketAdversary(faulty_fraction=0.5, view_soc_error=0.5, seed=2)
    w = _adv_world(auction_scheduler, adv)
    m = w.metrics()
    assert m["n_faulty_sats"] == 2
    assert m["view_corruptions_total"] > 0


def test_corruption_defeats_the_solvency_rule():
    """The mechanism, asserted directly: a wrong belief buys an illegal bid.

    Belief is corrupted; physics is NOT. In this eclipse-and-low-SOC world the
    honest auction refuses to start anything at all -- the reserve price is doing
    its job. The corrupted satellites believe they are charged, bid, win, and
    then starve on jobs they cannot power. That gap is the whole attack, and it
    is why the hook rewrites the VIEW rather than `self.soc`: had it written the
    world's state, the satellite would be genuinely charged, the run would look
    healthy, and the "attack" would be a relabelled outcome.
    """
    adv = MarketAdversary(faulty_fraction=0.5, view_soc_error=0.9, seed=2)
    honest = _adv_world(auction_scheduler, None).metrics()
    attacked = _adv_world(auction_scheduler, adv).metrics()

    assert honest["jobs_started"] == 0, \
        "the honest solvency rule should refuse every bid here; if it does not, " \
        "this world is not scarce and the test cannot detect the attack"
    assert attacked["jobs_started"] > 0, "corruption bought no illegal assignment"
    assert attacked["starved_job_steps"] > honest["starved_job_steps"]
    assert attacked["brownout_sat_steps"] > honest["brownout_sat_steps"]


def test_the_same_adversary_is_deterministic():
    adv = MarketAdversary(faulty_fraction=0.5, view_soc_error=0.5, seed=7)
    a = _adv_world(auction_scheduler, adv).metrics()
    b = _adv_world(auction_scheduler, adv).metrics()
    assert a == b


def test_a_fraction_that_marks_nobody_refuses():
    """Guardrail 4: a corruption that cannot fire reads as robustness."""
    with pytest.raises(ValueError, match="ZERO"):
        MarketAdversary(faulty_fraction=0.001, view_soc_error=0.4).marked(81)


def test_verify_refuses_a_view_attack_that_never_fired():
    adv = MarketAdversary(faulty_fraction=0.5, view_soc_error=0.4)
    with pytest.raises(NoCorruptionFiredError):
        verify_corruption_fired({"view_corruptions_total": 0}, adv)


def test_verify_refuses_a_bid_attack_on_a_scheduler_that_takes_no_bids():
    """FCFS is immune BY CONSTRUCTION, which is not a robustness result."""
    liar = MarketAdversary(faulty_fraction=0.5, bid_inflation=3.0)
    with pytest.raises(NoCorruptionFiredError, match="no bid was corrupted"):
        verify_corruption_fired({"view_corruptions_total": 0}, liar, {})


def test_an_honest_liar_reduces_to_the_plain_auction():
    """inflation=1.0 must be the honest auction exactly.

    This is what proves `lying_auction_scheduler` shares one clearing rule with
    `auction_scheduler` rather than carrying a divergent copy -- the trap
    `_utilities` already closes between the auction and CP-SAT.
    """
    honest = MarketAdversary(faulty_fraction=0.5, bid_inflation=1.0)
    assert not honest.corrupts_bids
    a = _adv_world(auction_scheduler).metrics()
    b = _adv_world(lambda v: lying_auction_scheduler(v, honest)).metrics()
    assert a == b


def test_bid_inflation_changes_the_clearing():
    liar = MarketAdversary(faulty_fraction=0.5, bid_inflation=50.0, seed=1)
    fired = {}
    _adv_world(lambda v: lying_auction_scheduler(v, liar, fired))
    assert fired["bid_corruptions_total"] > 0
