"""
Intent-CRDT with Contact-Plan DTN (ICCD).

The control plane that keeps mission *intent* coherent across a swarm that is
only intermittently connected. Three pieces:

* :class:`IntentCRDT` — mission intent as a composite CRDT (a last-writer-wins
  goal register, an observed-remove set of constraints, and a vector-clock
  causal context). This is the object whose convergence the ICCD paper proves
  in Appendix A; here it is built directly on :mod:`gossamer.crdt`, so the same
  algebra the unit tests exercise backs the propagation logic.
* :class:`ContactPlan` — scheduled communication windows. Connectivity is a
  function of time, not a static graph.
* :func:`dtn_sync_round` / :func:`select_relays` — propagate intent along active
  contacts with custody (merge on contact, never lose an update) and rotate
  relays by an energy x centrality score (paper Appendix B).

Because intents are CRDTs and the contact plan is eventually connected, every
agent converges to the same intent — strong eventual consistency without a
coordinator. :func:`dtn_sync_round` also tracks age-of-information (AoI), the
paper's headline freshness metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from gossamer.crdt import CompositeCRDT, LWWRegister, ORSet, VectorClock


# --------------------------------------------------------------------------
# Mission intent as a composite CRDT
# --------------------------------------------------------------------------

class IntentCRDT:
    """Mission intent: ``{goal: LWW, constraints: OR-Set, clock: VectorClock}``.

    All mutators return a *new* IntentCRDT; :meth:`merge` is the semilattice
    join inherited componentwise from the composite (ICCD Appendix A, Lemma 1).
    """

    def __init__(self, composite: Optional[CompositeCRDT] = None):
        self._c = composite or CompositeCRDT({
            "goal": LWWRegister(),
            "constraints": ORSet(),
            "clock": VectorClock(),
        })

    def set_goal(self, goal, timestamp: float, replica) -> "IntentCRDT":
        comps = dict(self._c.components)
        comps["goal"] = comps["goal"].set(goal, timestamp, replica)
        comps["clock"] = comps["clock"].tick(replica)
        return IntentCRDT(CompositeCRDT(comps))

    def add_constraint(self, constraint, tag) -> "IntentCRDT":
        comps = dict(self._c.components)
        comps["constraints"] = comps["constraints"].add(constraint, tag)
        return IntentCRDT(CompositeCRDT(comps))

    def remove_constraint(self, constraint) -> "IntentCRDT":
        comps = dict(self._c.components)
        comps["constraints"] = comps["constraints"].remove(constraint)
        return IntentCRDT(CompositeCRDT(comps))

    def merge(self, other: "IntentCRDT") -> "IntentCRDT":
        return IntentCRDT(self._c.merge(other._c))

    def value(self) -> Dict:
        return self._c.value()

    def goal(self):
        return self._c.components["goal"].value()

    def constraints(self) -> frozenset:
        return self._c.components["constraints"].value()


# --------------------------------------------------------------------------
# Contact plan
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Contact:
    u: int
    v: int
    start: float
    end: float

    def active_at(self, t: float) -> bool:
        return self.start <= t < self.end


class ContactPlan:
    """A set of scheduled (bidirectional) contact windows."""

    def __init__(self, contacts: Sequence[Contact]):
        self.contacts: List[Contact] = list(contacts)

    def active(self, t: float) -> List[Contact]:
        return [c for c in self.contacts if c.active_at(t)]

    def neighbors(self, node: int, t: float) -> Set[int]:
        out: Set[int] = set()
        for c in self.contacts:
            if not c.active_at(t):
                continue
            if c.u == node:
                out.add(c.v)
            elif c.v == node:
                out.add(c.u)
        return out

    def next_contact(self, node: int, t: float) -> Optional[Contact]:
        upcoming = [c for c in self.contacts
                    if (c.u == node or c.v == node) and c.start >= t]
        return min(upcoming, key=lambda c: c.start) if upcoming else None


# --------------------------------------------------------------------------
# Energy-aware relay selection (paper Appendix B)
# --------------------------------------------------------------------------

def select_relays(
    candidates: Sequence[int],
    coverage: Dict[int, Set[int]],
    soc: Dict[int, float],
    degree: Dict[int, float],
    clustering: Dict[int, float],
    battery_capacity: float = 1.0,
) -> Set[int]:
    """Greedy minimal relay cover, ranked by centrality x state-of-charge.

    Score ``kappa = (1/(clustering+eps)) * degree`` approximates betweenness;
    multiplied by the SOC fraction it prefers well-charged, well-connected
    relays. We then add relays in score order until every target in the union
    of coverage sets is covered (set-cover heuristic).
    """
    eps = 1e-9
    targets: Set[int] = set().union(*coverage.values()) if coverage else set()
    scored = sorted(
        candidates,
        key=lambda n: (1.0 / (clustering.get(n, 0.0) + eps)) * degree.get(n, 0.0)
        * (soc.get(n, 0.0) / max(battery_capacity, eps)),
        reverse=True,
    )
    selected: Set[int] = set()
    covered: Set[int] = set()
    for n in scored:
        if covered >= targets:
            break
        gain = coverage.get(n, set()) - covered
        if gain:
            selected.add(n)
            covered |= gain
    return selected


# --------------------------------------------------------------------------
# DTN propagation
# --------------------------------------------------------------------------

def prioritize_bundle(
    deltas: Sequence[Tuple[object, float, float]], k: int
) -> List[object]:
    """Top-``k`` delta ids by freshness-gain per joule (paper §3.2 / §5 utility).

    ``deltas`` are ``(id, freshness_gain, energy_cost)`` triples.
    """
    ranked = sorted(deltas, key=lambda d: d[1] / max(d[2], 1e-9), reverse=True)
    return [d[0] for d in ranked[:k]]


def dtn_sync_round(
    intents: List[IntentCRDT],
    plan: ContactPlan,
    t: float,
    aoi: Optional[np.ndarray] = None,
    dt: float = 1.0,
) -> Tuple[List[IntentCRDT], Optional[np.ndarray]]:
    """One DTN step: merge intent across every active contact (custody transfer).

    Returns updated intents and, if ``aoi`` is supplied, updated
    age-of-information: a node that exchanges with a fresher peer takes the
    minimum AoI (freshness improves on contact); otherwise AoI ages by ``dt``.
    """
    new_intents = list(intents)
    new_aoi = None if aoi is None else aoi.astype(float) + dt  # age everyone first
    for c in plan.active(t):
        merged = new_intents[c.u].merge(new_intents[c.v])
        new_intents[c.u] = merged
        new_intents[c.v] = merged
        if new_aoi is not None:
            fresh = min(new_aoi[c.u], new_aoi[c.v])
            new_aoi[c.u] = fresh
            new_aoi[c.v] = fresh
    return new_intents, new_aoi
