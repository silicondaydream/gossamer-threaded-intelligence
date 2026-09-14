"""Tests for the CRDT intent-routing layer (N3's submission side).

The structure mirrors what the layer is for. The first group asserts the
LATTICE still holds once updates are shipped as deltas rather than as whole
replicas -- if that breaks, every convergence number in N3 is measuring the bug
rather than the geometry. The second group asserts the two places this module
can silently lie: a digest that claims more than it holds, and a per-peer log
credited with a delivery the link never made.
"""
from __future__ import annotations

import pytest

from gossamer.algorithms.coordination.intent_routing import (
    ADD, CLOCK_ENTRY_BITS, GOAL, NEWEST_FIRST, OLDEST_FIRST, REMOVE,
    ROUTING_POLICIES, UPDATE_BITS, DeltaLogPolicy, DigestPolicy,
    FullStatePolicy, IntentReplica, IntentUpdate, NoRelayPolicy, prioritised)
from gossamer.crdt import CompositeCRDT


# --------------------------------------------------------------------------
# the lattice survives being shipped as deltas
# --------------------------------------------------------------------------

def _stream(n_updates=12, n_nodes=4):
    """A deterministic update stream authored across several replicas."""
    reps = [IntentReplica(i) for i in range(n_nodes)]
    updates = []
    for k in range(n_updates):
        author = k % n_nodes
        if k % 3 == 0:
            u = reps[author].author_update(k, float(k), GOAL, ("g", k))
        elif k % 3 == 1:
            u = reps[author].author_update(k, float(k), ADD, ("c", k % 5))
        else:
            live = sorted(reps[author].intent.constraints())
            if live:
                u = reps[author].author_update(k, float(k), REMOVE, live[0])
            else:
                u = reps[author].author_update(k, float(k), ADD, ("c", k % 5))
        updates.append(u)
    return updates


def test_delta_application_is_order_independent():
    """Commutativity, on the deltas themselves rather than on whole replicas.

    The state-based proof in `test_crdt.py` covers merging replicas. What N3
    ships is a *set of deltas*, applied in whatever order the contact plan
    happens to deliver them, so the property that matters is that any
    permutation of one delta set lands on the same value.
    """
    updates = _stream()
    values = set()
    for perm in (updates, list(reversed(updates)),
                 sorted(updates, key=lambda u: (u.author, u.seq)),
                 sorted(updates, key=lambda u: -u.uid)):
        r = IntentReplica(99)
        for u in perm:
            r.apply(u)
        values.add(r.observable())
    assert len(values) == 1


def test_delta_application_is_idempotent():
    updates = _stream()
    a = IntentReplica(0)
    for u in updates:
        a.apply(u)
    once = a.observable()
    for u in updates * 3:
        a.apply(u)
    assert a.observable() == once


def test_incremental_state_equals_the_fold_of_its_held_set():
    """The fast path and the lattice must agree; `verify` is what says so."""
    updates = _stream(24, 5)
    by_uid = {u.uid: u for u in updates}
    r = IntentReplica(7)
    for u in updates[::2]:
        r.apply(u)
    r.verify(by_uid)          # raises if the two disagree


def test_a_remove_is_built_from_the_authors_view_so_add_wins_can_fire():
    """The property the OR-Set is chosen FOR, and the bug that deletes it.

    If a remove's tombstone set were computed from a merged/omniscient state it
    would always find every tag and always succeed, so `addwins_fraction` would
    be a flat zero and the paper would report that concurrent edits are not a
    problem. Here the remover has never seen B's add, so B's add survives.
    """
    a, b, c = IntentReplica(0), IntentReplica(1), IntentReplica(2)
    add_a = a.author_update(1, 0.0, ADD, ("c", 0))
    add_b = b.author_update(2, 0.0, ADD, ("c", 0))      # concurrent, same element

    c.apply(add_a)                                      # c has only A's add
    rem = c.author_update(3, 1.0, REMOVE, ("c", 0))
    assert rem.removed_tags == ((0, 1),), "the remover tombstoned a tag it never saw"

    everyone = IntentReplica(3)
    for u in (add_a, add_b, rem):
        everyone.apply(u)
    assert ("c", 0) in everyone.intent.constraints(), "add-wins did not fire"


def test_a_remove_that_observed_every_tag_does_remove():
    """The other half: add-wins is not a licence for removes never to work."""
    a, b = IntentReplica(0), IntentReplica(1)
    add_a = a.author_update(1, 0.0, ADD, ("c", 0))
    add_b = b.author_update(2, 0.0, ADD, ("c", 0))
    for u in (add_a, add_b):
        a.apply(u)
    rem = a.author_update(3, 1.0, REMOVE, ("c", 0))
    assert len(rem.removed_tags) == 2
    assert ("c", 0) not in a.intent.constraints()


def test_lww_ties_break_on_replica_id_not_on_arrival_order():
    """Two writes at the same instant must resolve the same way everywhere."""
    a, b = IntentReplica(0), IntentReplica(1)
    ua = a.author_update(1, 5.0, GOAL, ("g", 1))
    ub = b.author_update(2, 5.0, GOAL, ("g", 2))
    fwd, rev = IntentReplica(8), IntentReplica(9)
    fwd.apply(ua); fwd.apply(ub)
    rev.apply(ub); rev.apply(ua)
    assert fwd.observable()[0] == rev.observable()[0] == ("g", 2)


# --------------------------------------------------------------------------
# the digest must not claim more than it holds
# --------------------------------------------------------------------------

def test_digest_is_the_contiguous_prefix_not_the_maximum_held():
    """A gap must stall the digest, or the peer withholds the gap forever.

    Advertising the maximum sequence held would tell a peer "I have everything
    from author 0", the peer would send nothing, and the hole would be permanent
    -- silently, and while reading as a bandwidth saving.
    """
    a = IntentReplica(0)
    src = IntentReplica(5)
    u1 = src.author_update(1, 0.0, ADD, ("c", 1))
    u2 = src.author_update(2, 1.0, ADD, ("c", 2))
    u3 = src.author_update(3, 2.0, ADD, ("c", 3))
    a.apply(u1)
    a.apply(u3)                       # out of order: 2 is missing
    assert a.digest() == {5: 1}, "the digest jumped the gap"
    assert not a.covered_by(a.digest(), u2)
    a.apply(u2)
    assert a.digest() == {5: 3}


def test_digest_policy_offers_exactly_what_the_prefix_cannot_cover():
    src = IntentReplica(5)
    ups = [src.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 5)]
    by_uid = {u.uid: u for u in ups}
    snd, rcv = IntentReplica(0), IntentReplica(1)
    for u in ups:
        snd.apply(u)
    rcv.apply(ups[0]); rcv.apply(ups[1])

    pol = DigestPolicy()
    pol.reset(2)
    offer, ctrl, bits = pol.select(snd, rcv, by_uid, 0.0)
    assert offer == {ups[2].uid, ups[3].uid}
    assert ctrl == snd.digest_bits() + rcv.digest_bits() > 0
    assert bits == ups[2].bits + ups[3].bits


def test_delta_log_credits_delivery_not_the_offer():
    """Crediting an offer is how a policy convinces itself of a delivery the
    window truncated away. The sent-set must move only on `on_delivered`."""
    src = IntentReplica(5)
    ups = [src.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 4)]
    by_uid = {u.uid: u for u in ups}
    snd, rcv = IntentReplica(0), IntentReplica(1)
    for u in ups:
        snd.apply(u)

    pol = DeltaLogPolicy()
    pol.reset(2)
    offer, _, bits = pol.select(snd, rcv, by_uid, 0.0)
    assert offer == {u.uid for u in ups} and bits == sum(u.bits for u in ups)
    landed = {ups[0].uid}
    pol.on_delivered(snd, rcv, landed)             # only one actually landed
    again, _, again_bits = pol.select(snd, rcv, by_uid, 1.0)
    assert again == {ups[1].uid, ups[2].uid}, "a truncated offer was credited"
    # The weight is kept by subtraction rather than re-summed; it must agree
    # with the sum, or the bandwidth column drifts from the offer silently.
    assert again_bits == ups[1].bits + ups[2].bits


def test_no_relay_forwards_nothing_it_did_not_author():
    mine = IntentReplica(0)
    other = IntentReplica(1)
    u_mine = mine.author_update(1, 0.0, ADD, ("c", 1))
    u_theirs = other.author_update(2, 0.0, ADD, ("c", 2))
    mine.apply(u_theirs)
    by_uid = {1: u_mine, 2: u_theirs}
    pol = NoRelayPolicy()
    pol.reset(2)
    offer, ctrl, bits = pol.select(mine, IntentReplica(2), by_uid, 0.0)
    assert offer == {1} and ctrl == 0 and bits == u_mine.bits


def test_full_state_offers_everything_held():
    a = IntentReplica(0)
    src = IntentReplica(5)
    ups = [src.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 4)]
    for u in ups:
        a.apply(u)
    pol = FullStatePolicy()
    pol.reset(2)
    offer, ctrl, bits = pol.select(a, IntentReplica(1), {u.uid: u for u in ups}, 0.0)
    assert offer == a.held and ctrl == 0
    assert bits == a.held_bits == sum(u.bits for u in ups)


# --------------------------------------------------------------------------
# sizes and ordering
# --------------------------------------------------------------------------

def test_a_remove_costs_more_on_the_wire_than_a_plain_delta():
    """It carries its tombstones; a flat size would under-report the OR-Set."""
    src = IntentReplica(0)
    add = src.author_update(1, 0.0, ADD, ("c", 0))
    rem = src.author_update(2, 1.0, REMOVE, ("c", 0))
    assert add.bits == UPDATE_BITS
    assert rem.bits == UPDATE_BITS + CLOCK_ENTRY_BITS * len(rem.removed_tags)
    assert rem.bits > add.bits


def test_priority_orders_are_exact_reverses_on_distinct_timestamps():
    src = IntentReplica(0)
    ups = [src.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 6)]
    by_uid = {u.uid: u for u in ups}
    uids = [u.uid for u in ups]
    assert prioritised(uids, by_uid, NEWEST_FIRST) == list(reversed(
        prioritised(uids, by_uid, OLDEST_FIRST)))


def test_an_unknown_priority_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        DigestPolicy(priority="whatever")
    src = IntentReplica(0)
    u = src.author_update(1, 0.0, ADD, ("c", 0))
    with pytest.raises(ValueError):
        prioritised([1], {1: u}, "whatever")


def test_every_registered_policy_shares_the_select_contract():
    """A policy that silently returns the wrong shape would be caught only by a
    campaign producing plausible numbers, which is the house failure mode."""
    src = IntentReplica(5)
    ups = [src.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 4)]
    by_uid = {u.uid: u for u in ups}
    snd, rcv = IntentReplica(0), IntentReplica(1)
    for u in ups:
        snd.apply(u)
    for name, ctor in ROUTING_POLICIES.items():
        pol = ctor()
        pol.reset(2)
        offer, ctrl, bits = pol.select(snd, rcv, by_uid, 0.0)
        # A SET, because the caller delivers by set difference; a list here
        # would still work and would be quadratic in the horizon.
        assert isinstance(offer, (set, frozenset)), name
        assert all(u in by_uid for u in offer), name
        assert isinstance(ctrl, int) and ctrl >= 0, name
        # The weight a policy reports must be the weight of what it offered.
        # It is kept incrementally, so a drift here is silent and would be
        # visible only as a bandwidth column nobody can reproduce.
        assert bits == sum(by_uid[u].bits for u in offer), name
        pol.on_delivered(snd, rcv, offer)


def test_the_held_bit_total_tracks_the_held_set():
    """`held_bits` is maintained incrementally and read as the full-state
    offer's weight; if it drifts, every bandwidth number in N3 is wrong and
    nothing raises."""
    src = IntentReplica(5)
    ups = ([src.author_update(k, float(k), ADD, ("c", 0)) for k in range(1, 4)]
           + [src.author_update(4, 4.0, REMOVE, ("c", 0))])
    r = IntentReplica(0)
    for u in ups * 2:                       # duplicates must not double-count
        r.apply(u)
    assert r.held_bits == sum(u.bits for u in ups)
    assert r.held == {u.uid for u in ups}


def test_above_prefix_is_the_same_answer_as_filtering_the_whole_held_set():
    """The suffix lookup is an optimisation; it must agree with the definition.

    Written against a replica holding a GAP, because that is the case the
    bisect could get wrong while looking right on contiguous data.
    """
    src_a, src_b = IntentReplica(5), IntentReplica(6)
    ups = [src_a.author_update(k, float(k), ADD, ("c", k)) for k in range(1, 6)]
    ups += [src_b.author_update(10 + k, float(k), ADD, ("c", k)) for k in range(1, 4)]
    by_uid = {u.uid: u for u in ups}
    holder = IntentReplica(0)
    for u in ups:
        if u.seq != 3:                      # a hole in each author's sequence
            holder.apply(u)

    for dg in ({}, {5: 2}, {5: 5}, {5: 2, 6: 1}, {6: 9}, {7: 4}):
        offer, bits = holder.above_prefix(dg)
        want = {uid for uid in holder.held
                if not holder.covered_by(dg, by_uid[uid])}
        assert offer == want, dg
        assert bits == sum(by_uid[u].bits for u in offer), dg


def test_an_update_of_unknown_kind_raises_rather_than_merging_as_a_no_op():
    """A no-op delta merges cleanly and changes nothing, which is this stack's
    house bug: the campaign would report perfect convergence of nothing."""
    bad = IntentUpdate(uid=1, author=0, seq=1, t_gen=0.0, kind="nonsense",
                       payload=None)
    with pytest.raises(ValueError):
        bad.delta()


def test_a_delta_is_a_composite_with_the_same_component_keys():
    """`CompositeCRDT.merge` refuses mismatched keys, so a delta missing one
    would raise mid-campaign rather than at construction."""
    src = IntentReplica(0)
    for kind, payload in ((GOAL, ("g", 1)), (ADD, ("c", 1))):
        d = src.author_update(1, 0.0, kind, payload).delta()
        assert isinstance(d, CompositeCRDT)
        assert set(d.components) == {"goal", "constraints", "clock"}
