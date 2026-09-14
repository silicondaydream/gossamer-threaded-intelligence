"""CRDT intent routing: what a satellite sends a peer when a window opens (N3).

ICCD's :mod:`~gossamer.algorithms.coordination.iccd` layer merges whole replicas
on contact and calls it done. That is the state-based CvRDT corner: always
correct, never cheap, and it answers no engineering question because there is
nothing to choose. The choice a real DTN control plane makes is *which deltas to
put in this window*, and the three standard answers differ in what they need to
remember about the peer, not in what they eventually converge to:

============== ========================= ==========================
Policy         Per-peer state            Sends
============== ========================= ==========================
``no_relay``   none                      only self-authored updates
``full_state`` none                      everything held
``delta_log``  O(|peers| x |updates|)    what it has not sent *this peer*
``digest``     none (O(|peers|) ints)    what the peer's digest cannot cover
============== ========================= ==========================

All four converge on an eventually-connected plan, so *convergence* is not what
separates them -- the lattice laws guarantee it and measuring it again would be
hashing a tie. What separates them is the bill, which is why every policy
reports ``payload_bits`` beside its effect. **Measured over the N3 campaign
(2040 cells, 20 seeds): identical convergence in all 240 (spec, seed) groups,
and a 180x spread in traffic** -- full-state 22.85 Gbit, delta-log 18.82,
digest 0.13, with 99.9% of full-state's traffic redundant.

> **⚠️ That ranking INVERTS once a window cannot carry the offer, and the digest
> is the one that fails.** At 1e3 bit/s with freshness-first truncation the
> digest -- cheapest by 180x -- never fully replicates **86.3%** of updates,
> against delta-log's 11.4%. The cause is head-of-line blocking on its own
> abstraction: truncation drops the oldest deltas, that opens a gap in the
> receiver's contiguous prefix, the prefix stalls below the gap, and the peer is
> re-sent everything above it forever. Reversing to oldest-first takes the
> digest to **0.0%** and destroys both other policies (full-state 98.3%,
> delta-log 96.1%), because they have no prefix to fill and simply starve the
> newest intent behind the backlog. **Neither truncation order is correct in
> general, and the choice matters more than the protocol.**

Three properties are load-bearing and each one is a trap this module refuses.

**The digest is CONTIGUOUS-PREFIX, and therefore conservative.** A vector clock
can say "I have everything from author *a* up to sequence 7"; it cannot say "I
have a's 3 and 5 but not 4". A replica that took delta 5 out of order must
therefore advertise 2, and will be re-sent 3 and 5. Advertising 5 instead would
make the peer withhold 3 and 4 *forever*, which reads as a bandwidth saving and
is silent permanent divergence. :meth:`IntentReplica.digest` returns the prefix;
``redundant_bits`` counts what that costs, so the saving is reported honestly
rather than bought with a correctness hole.

**A remove is generated from the AUTHOR'S state, not from the object.**
``ORSet.remove`` tombstones the tags the removing replica *currently observes*,
so a remove issued by a satellite that has not yet heard about a concurrent add
does not remove it -- that is add-wins, and on a contact plan whose wait tail is
hundreds of seconds it is the normal case rather than a corner. Constructing a
remove delta from the merged oracle instead of the author would quietly delete
the effect this module exists to measure.

**Updates carry a STATED size, never a measured one.** ``UPDATE_BITS`` and
``CLOCK_ENTRY_BITS`` are declared wire sizes for a control-plane delta; nothing
here serialises a Python object and calls its ``len()`` a spacecraft bus
measurement. They are assumptions, they are named as assumptions, and every
bandwidth conclusion is reported as a RATIO against the window volume the link
budget computes so a reader can rescale it.
"""
from __future__ import annotations

from bisect import insort
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from gossamer.algorithms.coordination.iccd import IntentCRDT
from gossamer.crdt import CompositeCRDT, LWWRegister, ORSet, VectorClock

__all__ = [
    "IntentUpdate",
    "IntentReplica",
    "RoutingPolicy",
    "NoRelayPolicy",
    "FullStatePolicy",
    "DeltaLogPolicy",
    "DigestPolicy",
    "ROUTING_POLICIES",
    "UPDATE_BITS",
    "CLOCK_ENTRY_BITS",
    "GOAL",
    "ADD",
    "REMOVE",
    "NEWEST_FIRST",
    "OLDEST_FIRST",
    "PRIORITY_ORDERS",
    "prioritised",
]

#: Update kinds. ``GOAL`` writes the LWW register (a concurrent write DESTROYS
#: the loser); ``ADD``/``REMOVE`` move the OR-Set (a concurrent pair does not).
#: Both components live in one object on purpose -- same transport, same plan,
#: and only the lattice differs, which is what isolates the lattice's effect.
GOAL = "goal"
ADD = "add"
REMOVE = "remove"

#: Stated wire sizes. An intent delta is an id, an author, a sequence number, a
#: stamp and a small payload; 64 bytes is a generous flight-software encoding.
#: A digest entry is a replica id plus a counter. ASSUMPTIONS, not measurements
#: -- see the module docstring.
UPDATE_BITS = 512
CLOCK_ENTRY_BITS = 48

#: Truncation orders. Which one a policy uses is a submission choice with a
#: measured cost; see :func:`prioritised`.
NEWEST_FIRST = "newest_first"
OLDEST_FIRST = "oldest_first"
PRIORITY_ORDERS = (NEWEST_FIRST, OLDEST_FIRST)


@dataclass(frozen=True)
class IntentUpdate:
    """One authored change to the shared mission intent, as a CRDT delta.

    ``seq`` is the author-local sequence number (1-based) and is the update's
    vector-clock coordinate, so ``(author, seq)`` is both its identity and its
    causal position. ``uid`` is a global counter used only for bookkeeping --
    nothing in the merge semantics reads it, and nothing may start to, or the
    result would depend on generation order rather than on the lattice.

    ``removed_tags`` is populated only for ``REMOVE`` and holds exactly the tags
    the author observed at generation time. That is the add-wins mechanism; see
    the module docstring for why it must not be recomputed later.
    """

    uid: int
    author: int
    seq: int
    t_gen: float
    kind: str
    payload: Any
    removed_tags: Tuple[Tuple[Any, int], ...] = ()

    @property
    def tag(self) -> Tuple[int, int]:
        """The OR-Set tag for an ``ADD``: unique by construction."""
        return (self.author, self.seq)

    @property
    def bits(self) -> int:
        """Stated wire size. A remove carries its tombstones, so it is bigger."""
        return UPDATE_BITS + CLOCK_ENTRY_BITS * len(self.removed_tags)

    def delta(self) -> CompositeCRDT:
        """This update alone, as a composite delta that merges into any replica.

        The clock component carries only ``{author: seq}``: merging is pointwise
        max, so folding the deltas of a *causally contiguous* prefix reproduces
        the author's clock exactly, and folding an out-of-order subset produces
        a clock that over-states what is held. That over-statement is why
        :meth:`IntentReplica.digest` reads the contiguous prefix from the held
        set instead of reading the merged clock.
        """
        goal = LWWRegister()
        constraints = ORSet()
        if self.kind == GOAL:
            goal = goal.set(self.payload, self.t_gen, self.author)
        elif self.kind == ADD:
            constraints = constraints.add(self.payload, self.tag)
        elif self.kind == REMOVE:
            constraints = ORSet(adds=frozenset(), removed=frozenset(self.removed_tags))
        else:  # pragma: no cover - constructed only by the generator below
            raise ValueError(f"unknown update kind {self.kind!r}")
        return CompositeCRDT({
            "goal": goal,
            "constraints": constraints,
            "clock": VectorClock({self.author: self.seq}),
        })


class IntentReplica:
    """One satellite's replica of the shared intent, plus what it holds.

    The CRDT value and the held-update set are maintained together and must
    agree: ``intent`` is the fold of every delta in ``held``. The set is not a
    convenience index -- it is what the digest and the delta policies read, and
    :func:`~gossamer.algorithms.coordination.intent_routing.IntentReplica.verify`
    re-folds from scratch to assert the two have not drifted.
    """

    def __init__(self, node: int) -> None:
        self.node = int(node)
        self.intent = IntentCRDT()
        self.held: Set[int] = set()
        #: Wire size of the whole held set, maintained incrementally. A
        #: full-state offer is the held set, and re-summing it per window is
        #: O(|held|) on the hottest path in the campaign -- quadratic in the run
        #: length for a quantity that only ever grows by one term.
        self.held_bits = 0
        #: Updates this replica authored. Cached because `no_relay` asks for it
        #: once per open window and it changes only on authoring.
        self.authored: Set[int] = set()
        self.authored_bits = 0
        #: author -> sorted list of sequence numbers held, for the prefix digest.
        self._seqs: Dict[int, Set[int]] = {}
        #: author -> ``(seq, uid, bits)`` sorted by seq. This is what makes a
        #: digest offer a SUFFIX lookup rather than a scan of the held set: a
        #: contiguous prefix covers exactly the entries at or below it, so
        #: everything the peer might be missing is one bisect away. Scanning
        #: instead is O(|held|) per open window, i.e. quadratic in the horizon.
        self._by_author: Dict[int, List[Tuple[int, int, int]]] = {}
        #: uid -> wire size, for the per-peer accounting the delta log keeps.
        self._bits: Dict[int, int] = {}
        self._local_seq = 0

    # -- authoring ---------------------------------------------------------

    def author_update(self, uid: int, t: float, kind: str, payload: Any) -> IntentUpdate:
        """Create and locally apply an update authored by this replica.

        A ``REMOVE`` is built from THIS replica's observed tags (add-wins), so a
        satellite that has not heard of a concurrent add cannot tombstone it.
        """
        self._local_seq += 1
        removed: Tuple[Tuple[Any, int], ...] = ()
        if kind == REMOVE:
            orset = self.intent._c.components["constraints"]
            removed = tuple(sorted(
                (t_ for (e, t_) in orset.adds
                 if e == payload and t_ not in orset.removed),
                key=lambda x: (x[0], x[1])))
        upd = IntentUpdate(uid=uid, author=self.node, seq=self._local_seq,
                           t_gen=float(t), kind=kind, payload=payload,
                           removed_tags=removed)
        self.apply(upd)
        self.authored.add(upd.uid)
        self.authored_bits += upd.bits
        return upd

    # -- receiving ---------------------------------------------------------

    def apply(self, upd: IntentUpdate) -> bool:
        """Merge one delta. Returns False if already held (merge is idempotent,
        so re-applying is harmless; the return value is for accounting)."""
        if upd.uid in self.held:
            return False
        self.held.add(upd.uid)
        self.held_bits += upd.bits
        self._seqs.setdefault(upd.author, set()).add(upd.seq)
        insort(self._by_author.setdefault(upd.author, []),
               (upd.seq, upd.uid, upd.bits))
        self._bits[upd.uid] = upd.bits
        self.intent = self.intent.merge(IntentCRDT(upd.delta()))
        return True

    # -- what it can tell a peer -------------------------------------------

    def digest(self) -> Dict[int, int]:
        """Contiguous-prefix vector clock: author -> highest seq held with no gap.

        Deliberately NOT the merged clock. See the module docstring: the merged
        clock reports the maximum seq held, which over a gap tells a peer to
        withhold updates this replica will then never receive.
        """
        out: Dict[int, int] = {}
        for author, seqs in self._seqs.items():
            k = 1
            while k in seqs:
                k += 1
            if k > 1:
                out[author] = k - 1
        return out

    def digest_bits(self) -> int:
        return CLOCK_ENTRY_BITS * len(self._seqs)

    def covered_by(self, digest: Dict[int, int], upd: IntentUpdate) -> bool:
        """True if ``digest`` proves the peer already holds ``upd``."""
        return digest.get(upd.author, 0) >= upd.seq

    def above_prefix(self, digest: Dict[int, int]) -> Tuple[Set[int], int]:
        """Everything held that ``digest`` does not prove the peer has, and its bits.

        A contiguous prefix covers exactly the entries at or below it, so this is
        a per-author suffix rather than a filter over the whole held set.
        """
        offer: Set[int] = set()
        bits = 0
        for author, entries in self._by_author.items():
            k = digest.get(author, 0)
            lo, hi = 0, len(entries)
            while lo < hi:                       # bisect_left on (k + 1, ...)
                mid = (lo + hi) // 2
                if entries[mid][0] <= k:
                    lo = mid + 1
                else:
                    hi = mid
            for _seq, uid, b in entries[lo:]:
                offer.add(uid)
                bits += b
        return offer, bits

    def bits_of(self, uid: int) -> int:
        """Wire size of one held update. Raises for one this replica never took,
        rather than returning a default a bandwidth column would absorb."""
        return self._bits[uid]

    # -- observation --------------------------------------------------------

    def observable(self) -> Tuple[Any, frozenset]:
        """What this replica would ACT on: the goal value and the live constraints.

        The clock is excluded on purpose -- two replicas holding different
        superseded goal writes are observably identical and behave identically,
        and reporting them as divergent would inflate every divergence number
        with differences no satellite can act on. State-level difference is
        reported separately, by comparing held sets.
        """
        return (self.intent.goal(), self.intent.constraints())

    def clock(self) -> VectorClock:
        return self.intent._c.components["clock"]

    def verify(self, updates: Dict[int, IntentUpdate]) -> None:
        """Re-fold the held set from scratch and assert it reproduces ``intent``.

        The incremental path and the lattice fold must agree; if they ever do
        not, every convergence number in the campaign is measuring the bug.
        """
        ref = IntentCRDT()
        for uid in sorted(self.held):
            ref = ref.merge(IntentCRDT(updates[uid].delta()))
        if (ref.goal(), ref.constraints()) != self.observable():
            raise AssertionError(
                f"replica {self.node} incremental state disagrees with the fold "
                f"of its own held set ({len(self.held)} updates)")


# --------------------------------------------------------------------------
# Routing policies -- the submission
# --------------------------------------------------------------------------

class RoutingPolicy:
    """What a sender offers a receiver when a window is open.

    :meth:`select` returns ``(offer, control_bits, payload_bits)``: the set of
    updates to transmit, the CONTROL bits spent deciding (a digest exchange is
    not free) and what the offer weighs. The policy reports its own weight
    because only the policy knows how to get it cheaply -- a full-state offer is
    the whole held set and re-summing it per window is quadratic in the run
    length, which is the difference between a campaign that runs in an hour and
    one that does not.

    The offer is a SET, and the caller relies on that: delivery is
    ``offer - receiver.held``, a C-level set difference, rather than a
    per-element membership test over an offer that grows all run.

    The caller owns the window capacity: it truncates the offer to what fits and
    reports back through :meth:`on_delivered` with what actually landed, so a
    policy that keeps per-peer state cannot record a delivery the link never
    made.
    """

    name = "abstract"
    #: Per-peer bookkeeping in updates x peers, as an order of magnitude. Part
    #: of the result: the cheapest policy on the wire is not the cheapest to run.
    per_peer_state = False

    def __init__(self, priority: str = NEWEST_FIRST) -> None:
        if priority not in PRIORITY_ORDERS:
            raise ValueError(f"unknown priority {priority!r}; expected one of "
                             f"{sorted(PRIORITY_ORDERS)}")
        self.priority = priority

    def reset(self, n: int) -> None:
        """Called once per run. ``n`` is the replica count."""

    def select(self, sender: IntentReplica, receiver: IntentReplica,
               updates: Dict[int, IntentUpdate],
               t: float) -> Tuple[Set[int], int, int]:
        raise NotImplementedError

    def on_delivered(self, sender: IntentReplica, receiver: IntentReplica,
                     uids: Sequence[int]) -> None:
        """Record what actually landed. Default: nothing to remember."""


class NoRelayPolicy(RoutingPolicy):
    """Send only what this satellite authored itself: the no-DTN floor.

    Intent then reaches exactly the satellites the author meets directly within
    the horizon. On a sparse plan that is a minority of the constellation, and
    the permanently-ignorant fraction is the measured argument for relaying at
    all -- the same role ``no_relay`` plays in ``contact-plan-aoi``.
    """

    name = "no_relay"

    def select(self, sender, receiver, updates, t):
        return (sender.authored, 0, sender.authored_bits)


class FullStatePolicy(RoutingPolicy):
    """Offer the entire held set every window: the state-based CvRDT corner.

    Correct by construction and stateless about peers. The receiver discards
    what it already has, so the *effect* is identical to the delta policies --
    only the bill differs, and it grows with the run rather than with the news.
    """

    name = "full_state"

    def select(self, sender, receiver, updates, t):
        return (sender.held, 0, sender.held_bits)


class DeltaLogPolicy(RoutingPolicy):
    """Send what this sender has not already sent THIS peer.

    Exact and minimal on the wire, and it pays for that with O(peers x updates)
    of per-node memory -- the cost that does not appear in a bandwidth column.
    The sent-set is updated from what was DELIVERED, never from what was
    offered: crediting a truncated offer is how a policy convinces itself a peer
    holds an update that the window dropped.
    """

    name = "delta_log"
    per_peer_state = True

    def reset(self, n: int) -> None:
        self._sent: Dict[Tuple[int, int], Set[int]] = {}
        self._sent_bits: Dict[Tuple[int, int], int] = {}


    def select(self, sender, receiver, updates, t):
        key = (sender.node, receiver.node)
        sent = self._sent.setdefault(key, set())
        offer = sender.held - sent
        # The sent set is a subset of the sender's held set by construction --
        # only delivered uids are credited, and the sender held every one it
        # delivered -- so the offer's weight is a subtraction rather than a sum
        # over an offer that is large for the whole first orbit.
        return (offer, 0, sender.held_bits - self._sent_bits.get(key, 0))

    def on_delivered(self, sender, receiver, uids):
        key = (sender.node, receiver.node)
        sent = self._sent.setdefault(key, set())
        bits = self._sent_bits.get(key, 0)
        for uid in uids:
            if uid not in sent:
                sent.add(uid)
                bits += sender.bits_of(uid)
        self._sent_bits[key] = bits


class DigestPolicy(RoutingPolicy):
    """Exchange contiguous-prefix digests, then send what the digest cannot cover.

    Stateless about peers -- the peer describes itself each window -- at the cost
    of a round trip inside the window and of re-sending anything past a gap in
    the peer's prefix. Both costs are reported: ``control_bits`` for the digest,
    ``redundant`` for updates the receiver turned out to already hold.
    """

    name = "digest"

    def select(self, sender, receiver, updates, t):
        dg = receiver.digest()
        offer, bits = sender.above_prefix(dg)
        # The digest crosses the link in both directions before any payload does.
        return (offer, receiver.digest_bits() + sender.digest_bits(), bits)


#: name -> zero-arg constructor. The registry a campaign sweeps.
ROUTING_POLICIES = {
    NoRelayPolicy.name: NoRelayPolicy,
    FullStatePolicy.name: FullStatePolicy,
    DeltaLogPolicy.name: DeltaLogPolicy,
    DigestPolicy.name: DigestPolicy,
}


def prioritised(uids: Iterable[int], updates: Dict[int, IntentUpdate],
                order: str = NEWEST_FIRST) -> List[int]:
    """Order an offer for a window that cannot carry all of it.

    This is a property of the SUBMISSION, not of the link, and it is swept
    because the two obvious answers fail in opposite ways:

    * ``newest_first`` is what a freshness objective asks for, and it starves
      the oldest un-delivered deltas indefinitely. Under a contiguous-prefix
      digest that starvation is head-of-line blocking: the prefix stalls at the
      gap, so the peer keeps advertising a clock from before it and keeps being
      re-sent everything after it, forever.
    * ``oldest_first`` fills the prefix and pays for it in freshness: the newest
      intent waits behind the backlog.

    Neither is correct in general; which one a constellation wants is decided by
    whether it is optimising age or convergence, and those are different
    objectives (see the environment docstring).
    """
    if order == NEWEST_FIRST:
        return sorted(uids, key=lambda u: (-updates[u].t_gen, -u))
    if order == OLDEST_FIRST:
        return sorted(uids, key=lambda u: (updates[u].t_gen, u))
    raise ValueError(f"unknown priority order {order!r}")
