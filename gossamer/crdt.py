"""
Conflict-free Replicated Data Types (CRDTs) for eventually-consistent
swarm state.

This module is the shared substrate behind three Arboria papers:

* **ICCD** — mission *intent* is a composite of a last-writer-wins register
  (scalar goal fields), an observed-remove set (constraint membership), and
  a vector-clock causal context. See :class:`CompositeCRDT`.
* **DMB + TF-ACO** — stigmergic *pheromone* deposits are grow-only counters
  (:class:`GCounter`) with an evaporation stamp.
* **HMA** — *depot inventory* is a positive/negative counter
  (:class:`PNCounter`).

Every type here is a state-based CRDT (a CvRDT): its state forms a
join-semilattice, so :meth:`merge` is **commutative, associative, and
idempotent**. Replicas that exchange state in *any* order, with arbitrary
duplication, converge to the same value once they have seen the same set of
updates ("strong eventual consistency"). The unit tests assert these
algebraic laws directly; they are the empirical backing for Theorem 1 of the
ICCD paper.

Design notes
------------
* **Pure Python, no NumPy.** These are control-plane structures, not array
  math; keeping them dependency-free means ``import gossamer.crdt`` is cheap
  and usable on embedded / flight-software targets.
* **Deterministic.** Nothing here reads a wall clock or an RNG. Timestamps
  and unique tags are supplied by the caller, so behaviour is fully
  controlled by the experiment seed tree and tests are reproducible.
* **Immutable merges.** Mutating operations (``set``, ``add``, ``increment``,
  …) and :meth:`merge` return *new* instances and never mutate in place. This
  makes the lattice laws easy to test and reason about. For hot loops where
  allocation matters, callers can keep a single replica and fold deltas into
  it with ``replica = replica.merge(delta)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, Hashable, Iterable, Mapping, Tuple

__all__ = [
    "CRDT",
    "VectorClock",
    "LWWRegister",
    "GCounter",
    "PNCounter",
    "ORSet",
    "CompositeCRDT",
    "merge_all",
    "converged",
]

ReplicaId = Hashable
Tag = Tuple[ReplicaId, int]


class CRDT:
    """Abstract base for state-based CRDTs.

    Subclasses implement :meth:`merge` (the semilattice join) and
    :meth:`value` (the observable read). ``merge`` must be commutative,
    associative, and idempotent; :func:`gossamer.crdt.converged` and the test
    suite check this.
    """

    def merge(self, other: "CRDT") -> "CRDT":  # pragma: no cover - interface
        raise NotImplementedError

    def value(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class VectorClock(CRDT):
    """Vector clock: per-replica logical counters with pointwise-max merge.

    Provides the causal context that orders updates in the composite intent
    object. The merge is the pointwise maximum, which is the canonical
    join-semilattice on ``ReplicaId -> int`` maps.
    """

    counts: Mapping[ReplicaId, int] = field(default_factory=dict)

    def tick(self, replica: ReplicaId) -> "VectorClock":
        """Return a new clock with ``replica`` advanced by one."""
        new = dict(self.counts)
        new[replica] = new.get(replica, 0) + 1
        return VectorClock(new)

    def merge(self, other: "VectorClock") -> "VectorClock":
        if not isinstance(other, VectorClock):
            raise TypeError("can only merge VectorClock with VectorClock")
        merged = dict(self.counts)
        for r, c in other.counts.items():
            if c > merged.get(r, 0):
                merged[r] = c
        return VectorClock(merged)

    def value(self) -> Dict[ReplicaId, int]:
        return dict(self.counts)

    def dominates(self, other: "VectorClock") -> bool:
        """True if ``self`` is greater-than-or-equal to ``other`` everywhere."""
        for r, c in other.counts.items():
            if self.counts.get(r, 0) < c:
                return False
        return True

    def compare(self, other: "VectorClock") -> str:
        """Return ``'eq'``, ``'lt'``, ``'gt'`` or ``'concurrent'``."""
        ge = self.dominates(other)  # self >= other everywhere
        le = other.dominates(self)  # other >= self everywhere
        if ge and le:
            return "eq"
        if ge:
            return "gt"
        if le:
            return "lt"
        return "concurrent"


@dataclass(frozen=True)
class LWWRegister(CRDT):
    """Last-writer-wins register.

    Each write carries a ``stamp = (timestamp, replica_id)``. Merge keeps the
    value with the lexicographically-greater stamp, so ties between equal
    timestamps are broken deterministically by ``replica_id``. An unset
    register has ``stamp = (float('-inf'), '')`` and loses every merge.
    """

    val: Any = None
    timestamp: float = float("-inf")
    replica: ReplicaId = ""

    @property
    def stamp(self) -> Tuple[float, Any]:
        return (self.timestamp, self.replica)

    def set(self, val: Any, timestamp: float, replica: ReplicaId) -> "LWWRegister":
        """Return a new register written by ``replica`` at ``timestamp``."""
        return LWWRegister(val=val, timestamp=float(timestamp), replica=replica)

    def merge(self, other: "LWWRegister") -> "LWWRegister":
        if not isinstance(other, LWWRegister):
            raise TypeError("can only merge LWWRegister with LWWRegister")
        # Compare (timestamp, replica) lexicographically; keep the larger.
        if other.stamp > self.stamp:
            return other
        return self

    def value(self) -> Any:
        return self.val


@dataclass(frozen=True)
class GCounter(CRDT):
    """Grow-only counter.

    State is a per-replica map of monotonically increasing counts; the value
    is their sum, and merge is the pointwise maximum. Used for TF-ACO
    pheromone deposits (deposits only ever accumulate; evaporation is handled
    by a separate stamp in the TF-ACO layer).
    """

    counts: Mapping[ReplicaId, int] = field(default_factory=dict)

    def increment(self, replica: ReplicaId, amount: int = 1) -> "GCounter":
        if amount < 0:
            raise ValueError("GCounter increments must be non-negative; use PNCounter")
        new = dict(self.counts)
        new[replica] = new.get(replica, 0) + amount
        return GCounter(new)

    def merge(self, other: "GCounter") -> "GCounter":
        if not isinstance(other, GCounter):
            raise TypeError("can only merge GCounter with GCounter")
        merged = dict(self.counts)
        for r, c in other.counts.items():
            if c > merged.get(r, 0):
                merged[r] = c
        return GCounter(merged)

    def value(self) -> int:
        return sum(self.counts.values())


@dataclass(frozen=True)
class PNCounter(CRDT):
    """Positive/negative counter built from two grow-only counters.

    Supports increment *and* decrement (value = positives − negatives) while
    staying a join-semilattice. Used for HMA depot inventory, which both
    accrues (micro deposits) and drains (hauler pickups).
    """

    positives: GCounter = field(default_factory=GCounter)
    negatives: GCounter = field(default_factory=GCounter)

    def increment(self, replica: ReplicaId, amount: int = 1) -> "PNCounter":
        if amount < 0:
            return self.decrement(replica, -amount)
        return replace(self, positives=self.positives.increment(replica, amount))

    def decrement(self, replica: ReplicaId, amount: int = 1) -> "PNCounter":
        if amount < 0:
            return self.increment(replica, -amount)
        return replace(self, negatives=self.negatives.increment(replica, amount))

    def merge(self, other: "PNCounter") -> "PNCounter":
        if not isinstance(other, PNCounter):
            raise TypeError("can only merge PNCounter with PNCounter")
        return PNCounter(
            positives=self.positives.merge(other.positives),
            negatives=self.negatives.merge(other.negatives),
        )

    def value(self) -> int:
        return self.positives.value() - self.negatives.value()


@dataclass(frozen=True)
class ORSet(CRDT):
    """Observed-remove set (add-wins).

    Each ``add(element)`` attaches a globally-unique ``tag`` to the element; a
    ``remove(element)`` tombstones exactly the tags currently observed for
    that element. An element is present iff it has at least one add-tag that
    has not been removed. Concurrent add/remove resolves *add-wins*, because
    the concurrent add introduces a fresh tag the remove never observed.

    Tags must be unique across replicas; the convention is
    ``tag = (replica_id, local_counter)``.
    """

    adds: FrozenSet[Tuple[Any, Tag]] = field(default_factory=frozenset)
    removed: FrozenSet[Tag] = field(default_factory=frozenset)

    def add(self, element: Any, tag: Tag) -> "ORSet":
        """Add ``element`` with unique ``tag = (replica_id, counter)``."""
        return ORSet(adds=self.adds | {(element, tag)}, removed=self.removed)

    def remove(self, element: Any) -> "ORSet":
        """Tombstone every currently-observed tag for ``element``."""
        tags = {t for (e, t) in self.adds if e == element and t not in self.removed}
        if not tags:
            return self
        return ORSet(adds=self.adds, removed=self.removed | tags)

    def merge(self, other: "ORSet") -> "ORSet":
        if not isinstance(other, ORSet):
            raise TypeError("can only merge ORSet with ORSet")
        return ORSet(adds=self.adds | other.adds, removed=self.removed | other.removed)

    def __contains__(self, element: Any) -> bool:
        return any(e == element and t not in self.removed for (e, t) in self.adds)

    def value(self) -> FrozenSet[Any]:
        return frozenset(e for (e, t) in self.adds if t not in self.removed)


@dataclass(frozen=True)
class CompositeCRDT(CRDT):
    """Named product of CRDTs, merged componentwise.

    This is the ICCD intent object: a finite product of join-semilattices is
    itself a join-semilattice (ICCD Appendix A, Lemma 1), so merging a
    ``{"goal": LWWRegister, "constraints": ORSet, "clock": VectorClock}``
    componentwise inherits commutativity, associativity, and idempotence.

    Both operands must carry the same component keys with merge-compatible
    types.
    """

    components: Mapping[str, CRDT] = field(default_factory=dict)

    def merge(self, other: "CompositeCRDT") -> "CompositeCRDT":
        if not isinstance(other, CompositeCRDT):
            raise TypeError("can only merge CompositeCRDT with CompositeCRDT")
        if set(self.components) != set(other.components):
            raise ValueError(
                "composite operands must have identical component keys: "
                f"{sorted(self.components)} != {sorted(other.components)}"
            )
        merged = {k: v.merge(other.components[k]) for k, v in self.components.items()}
        return CompositeCRDT(merged)

    def value(self) -> Dict[str, Any]:
        return {k: v.value() for k, v in self.components.items()}


def merge_all(items: Iterable[CRDT]) -> CRDT:
    """Fold ``merge`` over a non-empty iterable of like-typed CRDTs."""
    it = iter(items)
    try:
        acc = next(it)
    except StopIteration as exc:  # pragma: no cover - defensive
        raise ValueError("merge_all requires at least one item") from exc
    for x in it:
        acc = acc.merge(x)
    return acc


def converged(replicas: Iterable[CRDT]) -> bool:
    """True if every replica, after merging all the others, reads the same value.

    A cheap end-to-end check of strong eventual consistency: merge the whole
    set into each replica (order-independent by the lattice laws) and confirm
    the observable values agree.
    """
    reps = list(replicas)
    if not reps:
        return True
    full = merge_all(reps)
    target = full.value()
    return all(r.merge(full).value() == target for r in reps)
