"""Tests for gossamer.algorithms.coordination.iccd."""
import numpy as np

from gossamer.algorithms.coordination.iccd import (
    Contact,
    ContactPlan,
    IntentCRDT,
    dtn_sync_round,
    prioritize_bundle,
    select_relays,
)


def test_intent_merge_newest_goal_and_union_constraints():
    a = IntentCRDT().set_goal("survey", 1.0, "r1").add_constraint("no_fly_A", ("r1", 1))
    b = IntentCRDT().set_goal("relay", 2.0, "r2").add_constraint("no_fly_B", ("r2", 1))
    m = a.merge(b)
    assert m.goal() == "relay"  # later timestamp wins
    assert m.constraints() == frozenset({"no_fly_A", "no_fly_B"})
    # Merge is commutative at the intent level.
    assert b.merge(a).value() == m.value()


def test_contact_plan_queries():
    plan = ContactPlan([Contact(0, 1, 0.0, 5.0), Contact(1, 2, 3.0, 8.0)])
    assert plan.neighbors(1, t=4.0) == {0, 2}
    assert plan.neighbors(0, t=6.0) == set()
    nxt = plan.next_contact(2, t=0.0)
    assert nxt is not None and nxt.start == 3.0


def test_select_relays_covers_all_targets():
    coverage = {0: {10, 11}, 1: {11, 12}, 2: {12, 13}, 3: {13, 10}}
    soc = {0: 0.9, 1: 0.2, 2: 0.9, 3: 0.2}
    degree = {0: 4, 1: 2, 2: 4, 3: 2}
    clustering = {0: 0.1, 1: 0.5, 2: 0.1, 3: 0.5}
    relays = select_relays([0, 1, 2, 3], coverage, soc, degree, clustering)
    covered = set().union(*(coverage[r] for r in relays))
    assert {10, 11, 12, 13} <= covered          # full cover
    assert 0 in relays and 2 in relays          # high-score relays chosen
    assert len(relays) <= 3                      # near-minimal


def test_prioritize_bundle_by_freshness_per_joule():
    deltas = [("a", 10.0, 1.0), ("b", 9.0, 0.5), ("c", 1.0, 1.0)]
    # ratios: a=10, b=18, c=1 -> b, a
    assert prioritize_bundle(deltas, k=2) == ["b", "a"]
    assert prioritize_bundle(deltas, k=1) == ["b"]


def test_dtn_reaches_strong_eventual_consistency():
    # Four agents, each with its own goal (ts = id) and constraint.
    intents = [
        IntentCRDT().set_goal(f"g{k}", float(k), k).add_constraint(f"c{k}", (k, 1))
        for k in range(4)
    ]
    # A repeating chain contact plan (DTN: connectivity varies over time).
    contacts = []
    for cycle in range(12):
        base = cycle * 3
        contacts += [
            Contact(0, 1, base + 0, base + 1),
            Contact(1, 2, base + 1, base + 2),
            Contact(2, 3, base + 2, base + 3),
        ]
    plan = ContactPlan(contacts)

    for t in range(36):
        intents, _ = dtn_sync_round(intents, plan, float(t))

    target = intents[0].value()
    assert all(it.value() == target for it in intents)          # all converged
    assert intents[0].goal() == "g3"                            # newest goal won
    assert intents[0].constraints() == frozenset({"c0", "c1", "c2", "c3"})


def test_dtn_improves_age_of_information():
    intents = [IntentCRDT().set_goal("g", float(k), k) for k in range(4)]
    plan = ContactPlan([
        Contact(0, 1, float(t), float(t) + 1) if t % 3 == 0 else
        Contact(1, 2, float(t), float(t) + 1) if t % 3 == 1 else
        Contact(2, 3, float(t), float(t) + 1)
        for t in range(30)
    ])
    aoi = np.array([0.0, 100.0, 100.0, 100.0])  # only node 0 is fresh initially

    connected = aoi.copy()
    control = aoi.copy()  # no contacts: ages only
    cur = list(intents)
    for t in range(30):
        cur, connected = dtn_sync_round(cur, plan, float(t), connected, dt=1.0)
        control = control + 1.0
    # Sharing freshness across contacts keeps mean AoI well below the
    # age-only control.
    assert connected.mean() < control.mean()
