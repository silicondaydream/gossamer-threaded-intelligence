"""
Tests for gossamer.crdt.

The bulk of these assert the three semilattice laws — commutativity,
associativity, idempotence — for every CRDT type, plus order-independent
convergence under randomly shuffled, duplicated delivery. Those laws are the
formal content of "strong eventual consistency" and the empirical backing for
Theorem 1 of the ICCD paper.
"""
import itertools
import random

import pytest

from gossamer.crdt import (
    CompositeCRDT,
    GCounter,
    LWWRegister,
    ORSet,
    PNCounter,
    VectorClock,
    converged,
    merge_all,
)


# --------------------------------------------------------------------------
# Generic lattice-law checks, parametrised over a set of sample states per type
# --------------------------------------------------------------------------

def _lww_samples():
    base = LWWRegister()
    return [
        base,
        base.set("a", 1.0, "r1"),
        base.set("b", 2.0, "r2"),
        base.set("c", 2.0, "r1"),  # same ts as r2, lower replica id -> loses
    ]


def _gcounter_samples():
    g = GCounter()
    return [
        g,
        g.increment("r1", 3),
        g.increment("r2", 5),
        g.increment("r1", 1).increment("r2", 2),
    ]


def _pncounter_samples():
    p = PNCounter()
    return [
        p,
        p.increment("r1", 5),
        p.decrement("r2", 2),
        p.increment("r1", 5).decrement("r1", 3),
    ]


def _orset_samples():
    s = ORSet()
    s1 = s.add("x", ("r1", 1))
    s2 = s.add("y", ("r2", 1)).add("x", ("r2", 2))
    s3 = s1.remove("x")
    return [s, s1, s2, s3]


def _vclock_samples():
    v = VectorClock()
    return [
        v,
        v.tick("r1").tick("r1"),
        v.tick("r2"),
        v.tick("r1").tick("r2"),
    ]


def _composite_samples():
    def make(reg_val, reg_ts, reg_rep, add_el, add_tag, tick_rep):
        return CompositeCRDT({
            "goal": LWWRegister().set(reg_val, reg_ts, reg_rep),
            "constraints": ORSet().add(add_el, add_tag),
            "clock": VectorClock().tick(tick_rep),
        })
    return [
        make("g1", 1.0, "r1", "c1", ("r1", 1), "r1"),
        make("g2", 2.0, "r2", "c2", ("r2", 1), "r2"),
        make("g3", 2.0, "r1", "c1", ("r3", 1), "r1"),
    ]


ALL_SAMPLE_SETS = {
    "lww": _lww_samples,
    "gcounter": _gcounter_samples,
    "pncounter": _pncounter_samples,
    "orset": _orset_samples,
    "vclock": _vclock_samples,
    "composite": _composite_samples,
}


@pytest.mark.parametrize("name", list(ALL_SAMPLE_SETS))
def test_commutative(name):
    samples = ALL_SAMPLE_SETS[name]()
    for a, b in itertools.product(samples, repeat=2):
        assert a.merge(b).value() == b.merge(a).value()


@pytest.mark.parametrize("name", list(ALL_SAMPLE_SETS))
def test_associative(name):
    samples = ALL_SAMPLE_SETS[name]()
    for a, b, c in itertools.product(samples, repeat=3):
        left = a.merge(b).merge(c).value()
        right = a.merge(b.merge(c)).value()
        assert left == right


@pytest.mark.parametrize("name", list(ALL_SAMPLE_SETS))
def test_idempotent(name):
    samples = ALL_SAMPLE_SETS[name]()
    for a in samples:
        assert a.merge(a).value() == a.value()


@pytest.mark.parametrize("name", list(ALL_SAMPLE_SETS))
def test_convergence_under_shuffled_duplicated_delivery(name):
    """Every replica converges regardless of delivery order or duplication."""
    samples = ALL_SAMPLE_SETS[name]()
    rng = random.Random(1234)
    target = merge_all(samples).value()
    for _ in range(50):
        delivery = list(samples)
        # Duplicate a few updates and shuffle: CRDTs must tolerate both.
        delivery += rng.sample(samples, k=rng.randint(0, len(samples)))
        rng.shuffle(delivery)
        assert merge_all(delivery).value() == target
    assert converged(samples)


# --------------------------------------------------------------------------
# Semantic checks specific to each type
# --------------------------------------------------------------------------

def test_lww_tiebreak_is_deterministic_by_replica():
    a = LWWRegister().set("from_r1", 5.0, "r1")
    b = LWWRegister().set("from_r2", 5.0, "r2")  # same ts, higher replica id wins
    assert a.merge(b).value() == "from_r2"
    assert b.merge(a).value() == "from_r2"


def test_lww_latest_timestamp_wins():
    a = LWWRegister().set("old", 1.0, "r1")
    b = LWWRegister().set("new", 9.0, "r1")
    assert a.merge(b).value() == "new"


def test_gcounter_sums_and_rejects_negative():
    g = GCounter().increment("r1", 3).increment("r2", 4)
    assert g.value() == 7
    with pytest.raises(ValueError):
        g.increment("r1", -1)


def test_gcounter_merge_takes_pointwise_max_not_sum():
    # Two replicas that independently saw r1's deposits must not double-count.
    a = GCounter().increment("r1", 5)
    b = a.increment("r1", 0)  # same observation of r1
    assert a.merge(b).value() == 5


def test_pncounter_inc_dec():
    p = PNCounter().increment("r1", 10).decrement("r2", 3)
    assert p.value() == 7
    assert p.increment("r1", -4).value() == 3  # negative increment -> decrement


def test_orset_add_wins_over_concurrent_remove():
    # r1 removes x while r2 concurrently re-adds x with a fresh tag -> x stays.
    r1 = ORSet().add("x", ("r1", 1))
    r2 = ORSet().add("x", ("r1", 1))  # both saw the original add
    r1 = r1.remove("x")
    r2 = r2.add("x", ("r2", 1))  # concurrent fresh add
    merged = r1.merge(r2)
    assert "x" in merged
    assert merged.value() == frozenset({"x"})


def test_orset_remove_after_observed_add():
    s = ORSet().add("x", ("r1", 1)).remove("x")
    assert "x" not in s
    assert s.value() == frozenset()


def test_vectorclock_ordering():
    v0 = VectorClock()
    a = v0.tick("r1")
    b = a.tick("r2")
    assert b.compare(a) == "gt"
    assert a.compare(b) == "lt"
    c = v0.tick("r2")
    assert a.compare(c) == "concurrent"
    assert a.compare(a) == "eq"


def test_composite_requires_matching_keys():
    a = CompositeCRDT({"goal": LWWRegister()})
    b = CompositeCRDT({"other": LWWRegister()})
    with pytest.raises(ValueError):
        a.merge(b)


def test_composite_merges_componentwise():
    a = CompositeCRDT({
        "goal": LWWRegister().set("survey", 1.0, "r1"),
        "constraints": ORSet().add("no_fly_A", ("r1", 1)),
        "clock": VectorClock().tick("r1"),
    })
    b = CompositeCRDT({
        "goal": LWWRegister().set("relay", 2.0, "r2"),  # newer -> wins
        "constraints": ORSet().add("no_fly_B", ("r2", 1)),
        "clock": VectorClock().tick("r2"),
    })
    merged = a.merge(b).value()
    assert merged["goal"] == "relay"
    assert merged["constraints"] == frozenset({"no_fly_A", "no_fly_B"})
    assert merged["clock"] == {"r1": 1, "r2": 1}


def test_type_mismatches_raise():
    with pytest.raises(TypeError):
        GCounter().merge(PNCounter())
    with pytest.raises(TypeError):
        LWWRegister().merge(ORSet())
