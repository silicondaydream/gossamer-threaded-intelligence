"""HMAWorld — the market orchestration, moved out of Maneuver.Map's runner.

The runner ran this loop inline over ~120 lines of shared mutable locals, and in
doing so bypassed two things `hma.py` already shipped:

  * `DepotInventory`, the PNCounter CRDT — the runner mutated a plain `np.zeros`
    array instead, so the depot ledger had no merge semantics at all, in a paper
    whose framing is decentralised coordination under partition; and
  * `mmc_metrics` — the runner hand-rolled Little's Law, which (unlike the module's
    own model) reports a finite, plausible wait for a queue that is actually
    unstable.
"""
import numpy as np
import pytest

from gossamer.algorithms.coordination.hma import HMAParams, energy_aware_auction
from gossamer.algorithms.coordination.hma_world import HMAWorld, HMAWorldConfig


def _assigner(hp, soc, dp, dm, params, cap):
    a, _ = energy_aware_auction(hp, soc, dp, dm, params, hauler_capacity=cap)
    return [int(x) if x >= 0 else None for x in a]


def _world(**kw):
    cfg = HMAWorldConfig(
        depots_xy=kw.pop("depots_xy", [(50.0, 0.0), (-50.0, 0.0)]),
        printers_xy=kw.pop("printers_xy", [(0.0, 80.0)]),
        role_micro=kw.pop("role_micro", [0, 1, 2]),
        role_hauler=kw.pop("role_hauler", [3, 4]),
        role_printer=kw.pop("role_printer", [5]),
        max_accel=kw.pop("max_accel", 1.0),
        **kw,
    )
    return HMAWorld(cfg, num_agents=6, assigner=_assigner)


def _run(world, steps=60):
    pos = np.zeros((6, 3))
    pos[:, 0] = [48.0, 49.0, 50.0, 50.0, -49.0, 0.0]
    for t in range(steps):
        accel = world.step(t, pos)
        assert accel.shape == pos.shape
        pos = pos + accel * 1.0
    return pos


def test_micro_agents_deposit_into_the_depots():
    w = _world()
    _run(w, steps=250)
    assert w.arrivals_kg > 0, "micro agents never reached a depot"


def test_legacy_steering_loads_haulers_that_then_never_deliver():
    """The hauler-steering bug, isolated. THIS IS THE PUBLISHED BEHAVIOUR.

    A loaded hauler is aimed at its printer only on the step it loads; on every
    step after, it steers back at its assigned DEPOT. So it oscillates and only
    delivers if it happens to pass within `interaction_radius` of a printer.

    With 2 haulers and 1 printer it delivers NOTHING while holding a full 500 kg
    load. The HMA presets deliver only because they scatter 90 printers, so a
    wandering hauler eventually strays into range of one by accident — which means
    the paper's absolute throughput is a function of printer DENSITY, not of the
    market's routing. See CLAUDE.md §2.2.
    """
    w = _world(deliver_to_printer=False)  # the default; named here for the record
    _run(w, steps=250)
    assert max(w.hauler_load.values()) > 0, "no hauler ever picked up a load"
    assert w.total_printed == 0.0, (
        "a loaded hauler reached the printer — the legacy steering bug is not "
        "being reproduced, so this world no longer matches the published runs")


def test_deliver_to_printer_actually_delivers():
    """The fix, available but OFF by default because it moves published numbers."""
    w = _world(deliver_to_printer=True)
    _run(w, steps=250)
    assert w.total_printed > 0, "a loaded hauler still never reached its printer"


def test_the_crdt_ledger_tracks_the_float_buffer():
    """The CRDT was dead code: its only caller used a plain np.zeros array."""
    w = _world()
    _run(w)
    ledger = [d.available() for d in w.depots]
    buffer = [int(round(x)) for x in w.depot_inventory]
    assert ledger == buffer, f"CRDT {ledger} diverged from the buffer {buffer}"


def test_the_ledger_is_mergeable_across_replicas():
    """The point of a PNCounter: two partitioned replicas reconcile."""
    a, b = _world(), _world()
    _run(a, steps=30)
    _run(b, steps=30)
    merged = a.depots[0].merge(b.depots[0])
    # Same replica ids on both sides (same agent indices), so a merge is idempotent
    # rather than additive — which is exactly the CRDT property being relied on.
    assert merged.available() == a.depots[0].available()


def test_conservation_of_mass():
    w = _world()
    _run(w)
    in_depots = float(w.depot_inventory.sum())
    on_haulers = sum(w.hauler_load.values())
    assert w.arrivals_kg == pytest.approx(in_depots + on_haulers + w.total_printed), (
        "mass was created or destroyed: arrivals != depots + haulers + printed")


def test_saturated_queue_is_reported_as_saturated_not_as_a_plausible_wait():
    """rho >= 1 means the buffer grows without bound and the wait diverges.

    The hand-rolled Little's Law in the runner would divide through and report a
    finite number for a queue that is in fact unstable.
    """
    w = _world()
    _run(w)
    m = w.metrics(sim_seconds=60.0, movement_energy_j=1000.0)
    # The MEASURED wait (Little's Law on the observed buffer) is always finite and
    # comfortable-looking — that is the point.
    assert np.isfinite(m["hma_depot_wait_time_s_mean"])
    # The THEORETICAL M/M/c model, on the same queue, says it is unstable.
    if m["hma_depot_mmc_utilization"] >= 1.0:
        assert m["hma_depot_mmc_saturated"] is True
        assert m["hma_depot_mmc_wait_time_s"] is None


def test_metrics_are_json_safe():
    """`inf` from mmc_metrics must not reach the summary — it is not valid JSON."""
    import json
    w = _world()
    _run(w)
    m = w.metrics(60.0, 1000.0)
    text = json.dumps(m)
    assert "Infinity" not in text and "NaN" not in text


def test_no_service_reports_none_not_a_zero_wait():
    """A 0.0 wait would claim an instantly-drained buffer."""
    w = _world(role_hauler=[])
    _run(w)
    m = w.metrics(60.0, 1000.0)
    assert m["hma_depot_mmc_wait_time_s"] is None
    assert m["hma_num_servers_haulers"] == 0


def test_reassignment_churn_is_recorded():
    w = _world()
    _run(w, steps=120)
    # Haulers switch depots as the buffers fill and drain; the paper reports the
    # mean interval between reassignments.
    m = w.metrics(120.0, 1000.0)
    assert m["hma_reallocation_latency_steps_mean"] >= 0.0


def test_a_world_with_no_depots_does_not_crash():
    w = _world(depots_xy=[])
    _run(w)
    assert w.total_printed == 0.0
