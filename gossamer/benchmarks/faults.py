"""Standardized fault models for the benchmark (roadmap 18).

A fault model is a named, frozen set of engine-config overrides plus **the metric
that proves it fired**. The second half is the whole design.

Why the evidence metric is not optional
---------------------------------------
A fault row is the single easiest place in this stack to ship a no-op, because
every failure mode of the row looks like a result:

* a fault that never fires reads as "the swarm was robust";
* a fault the engine silently ignores (wrong substrate, unsupported key) reads
  the same way;
* and both produce a clean, plausible leaderboard with every submission tied.

`ByzantineScenario` shipped exactly this — it marked adversaries and nothing read
the marks, so the "byzantine" row was plain rendezvous under a different label.
Two more have been caught since (a ground-station network invisible from the
constellation's inclination, a thermal node too slow to ever reach its limit).
So a fault model here MUST name a metric that is non-zero if and only if the fault
actually occurred, and the harness refuses the run when it comes back zero. A
model with nothing to point at cannot be added.

Substrate
---------
Faults live in the C++ engine. `ReferenceEngine` has no fault module at all, so a
fault model run against it would apply nothing and report a fault-free run —
precisely the silent no-op above. `requires_engine_faults` marks those models, and
the harness raises rather than running them on a substrate that cannot express
them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

__all__ = ["FaultModel", "FAULT_MODELS", "NoFaultsFiredError"]


class NoFaultsFiredError(RuntimeError):
    """A fault model was requested and the run shows no evidence it happened."""


@dataclass(frozen=True)
class FaultModel:
    """One named failure regime.

    ``evidence_metric`` is the engine metric that must be strictly positive after
    a run for the row to be reportable. ``None`` only for the fault-free control,
    which is the one case where "nothing happened" is the intended condition.
    """

    name: str
    doc: str
    config: Dict[str, str] = field(default_factory=dict)
    evidence_metric: Optional[str] = None
    requires_engine_faults: bool = True

    def verify(self, metrics: Dict[str, float]) -> float:
        """Return the evidence value, or raise if the fault left no trace."""
        if self.evidence_metric is None:
            return 0.0
        if self.evidence_metric not in metrics:
            raise NoFaultsFiredError(
                f"fault model {self.name!r} reports its evidence through "
                f"{self.evidence_metric!r}, which this engine does not expose. The run "
                f"cannot be shown to have contained any faults at all, so its result is "
                f"not reportable — it would read as robustness. Run on an engine that "
                f"implements faults (LeviathanEngine).")
        value = float(metrics[self.evidence_metric])
        if value <= 0.0:
            raise NoFaultsFiredError(
                f"fault model {self.name!r} fired ZERO faults over this run "
                f"({self.evidence_metric} = {value}). A fault-free run filed under a "
                f"fault label is a no-op that reads as a robustness result. Raise the "
                f"rate, lengthen the run, or use the 'none' control deliberately.")
        return value


# A bit flip's damage is concentrated in the high bits, and by a lot. Flipping bit b
# of an IEEE-754 double changes it by 2^(b-52) relative, so on a value of 100 the
# perturbation runs from 1.4e-14 (bit 0) to 32 (bit 51) — a MEDIAN over uniform
# mantissa bits of 7e-7, i.e. nothing. That is why the registry is a LADDER over bit
# ranges rather than one "radiation" row:
#
#   none  <=  seu (uniform mantissa)  <  seu_significant (top mantissa)  <<  seu_unprotected (exponent)
#
# and the spacing of that ladder is a quantitative argument about where ECC is worth
# spending: protecting the exponent buys almost everything, protecting the top of the
# mantissa buys the rest, protecting the low mantissa buys nothing measurable. Read
# `seu` alone and you would conclude a swarm is radiation-robust; read the ladder and
# you learn which bits that conclusion depends on.

#: The frozen registry. Rates are chosen to fire reliably over a benchmark-length
#: run WITHOUT saturating — a regime where every agent is dead or corrupt ties every
#: submission just as thoroughly as one where nothing happens (guardrail 4).
FAULT_MODELS: Dict[str, FaultModel] = {
    "none": FaultModel(
        name="none",
        doc="The fault-free control. Every other row is read against this one.",
        config={"fault_prob": "0.0", "seu_rate": "0.0"},
        evidence_metric=None,
        requires_engine_faults=False,
    ),
    "fail_stop": FaultModel(
        name="fail_stop",
        doc="Permanent, MARKED failure: the agent stops and everyone can see it. "
            "The classic model, and the easy one — peers can route around a corpse.",
        config={"fault_prob": "0.0005", "seu_rate": "0.0"},
        evidence_metric="num_faulty",
    ),
    "seu": FaultModel(
        name="seu",
        doc="Transient, SILENT corruption: radiation flips one UNIFORMLY-CHOSEN "
            "mantissa bit of one position/velocity word. The agent is not marked and "
            "does not stop — it is confidently wrong. Physically faithful for memory "
            "whose exponent and sign are protected, and it is expected to score close "
            "to `none`: see the bit-significance note below. That near-null is the "
            "measurement, not a broken row.",
        config={"fault_prob": "0.0", "seu_rate": "0.002",
                "seu_bit_low": "0", "seu_bit_high": "51"},
        evidence_metric="seu_flips_total",
    ),
    "seu_significant": FaultModel(
        name="seu_significant",
        doc="The same upset confined to the TOP of the mantissa (bits 45-51), where a "
            "flip is numerically significant. This isolates the damaging fraction of "
            "`seu` from the harmless bulk of it, which is what makes the ladder an "
            "argument about where to spend ECC rather than a single 'radiation' row.",
        config={"fault_prob": "0.0", "seu_rate": "0.002",
                "seu_bit_low": "45", "seu_bit_high": "51"},
        evidence_metric="seu_flips_total",
    ),
    "seu_unprotected": FaultModel(
        name="seu_unprotected",
        doc="The same upset rate in UNPROTECTED memory: the exponent is in range, so "
            "a flip replaces the value (→1e300) rather than nudging it. The contrast "
            "with `seu` is the measurable argument for ECC/scrubbing, which is why "
            "both ship rather than one 'radiation' row.",
        config={"fault_prob": "0.0", "seu_rate": "0.002",
                "seu_bit_low": "52", "seu_bit_high": "62"},
        evidence_metric="seu_flips_total",
    ),
}
