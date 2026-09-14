"""Criticality validation of the benchmark substrate (roadmap 18, the DMB absorption).

``vicsek_transition`` asks one question -- can this substrate order at all -- and
its docstring is explicit that it is deliberately NOT a criticality study:
exponents and finite-size scaling need an N-ladder and
:mod:`gossamer.metrics.criticality`, "and they belong to a paper, not a
smoke-check, which is exactly the line the archived DMB abstract crossed."

This module is that ladder, and the point of it is to give the DMB toolkit a
permanent, tested home pointed at a benchmark row rather than at a retired
paper's private script. The archived DMB abstract asserted 3-D Vicsek-class
exponents *from a custom boids engine*; what survives that is the validation
move, and a validation move is only worth anything if it is run.

What it computes, per ``(N, eta)`` and pooled over seeds:

* ``psi``  -- the steady-state polarisation, the order parameter.
* ``chi``  -- ``N * Var(psi)``, which must PEAK at the transition and whose peak
  must grow with N. A flat susceptibility means there is no transition here.
* ``U4``   -- the Binder cumulant. Curves for different N cross at ``eta_c``;
  that crossing is the standard evidence a transition is a genuine critical
  point rather than a crossover, and its ABSENCE is a result too.

FOUR THINGS THIS REFUSES TO DO, each because the obvious version is wrong
-------------------------------------------------------------------------
1. **The ladder is density-matched.** ``VicsekTransitionScenario(num_agents=N)``
   recomputes ``bound`` to hold the validated number density, because growing N
   at a fixed box densifies the swarm and the curve then measures density, not
   size -- and a finite-size scaling argument built on that is measuring the
   wrong axis. The engine's own scalability sweep follows the same rule.
2. **Fluctuations come from the run, not from the seeds.** ``chi`` is the
   variance of ``psi`` over the equilibrated TIME series, which is what the
   fluctuation-dissipation form means. Taking the variance over seed means
   instead measures the spread of independent experiments -- a different and
   usually larger number that peaks in the same place and would look fine.
   Both are reported (``chi``, ``chi_between_seed``) so the two cannot be
   confused, and the audit reports their ratio rather than picking one.
3. **The transient is discarded, and the discard is checked.** A Binder cumulant
   taken over a series that is still relaxing reports the relaxation. Every cell
   carries ``drift``, the difference in mean ``psi`` between the first and
   second half of the measured window; a cell whose drift is large has not
   equilibrated and is reported, never averaged in silently.
4. **A crossing is LOCATED, not asserted.** :func:`binder_crossing` returns the
   crossing for each consecutive size pair plus their spread. Reporting one
   number for "the" crossing hides whether the pairs agree, and pairs that do
   not agree are the signature of a crossover rather than a critical point.

This is NOT registered in ``ALL_SCENARIOS``. It consumes the existing
``vicsek_transition`` row; registering a new scenario would change what
``SUITE_VERSION`` contains and re-pin N2's published 2400-cell leaderboard,
which is a version bump and an owner's call.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from gossamer.benchmarks.baselines import vicsek_alignment
from gossamer.benchmarks.harness import (
    SUITE_VERSION, BenchmarkConfig, PhysicsEngine, run_benchmark)
from gossamer.benchmarks.scenarios import VicsekTransitionScenario
from gossamer.metrics.criticality import binder_cumulant, susceptibility

__all__ = ["CriticalitySpec", "run_cell", "run_ladder", "reduce_ladder",
           "binder_crossing",
           "susceptibility_peaks", "audit_ladder"]


@dataclass(frozen=True)
class CriticalitySpec:
    """The frozen ladder. Any change to it is a different measurement.

    ``eta`` is noise injected on the COMMAND, not on the heading angle: the
    scenario's ``corrupt_actions`` adds ``U(-eta, eta)`` to the acceleration the
    baseline returned, and the engine integrates it. So this is not numerically
    the same control parameter as textbook Vicsek's angular noise, and no
    exponent measured here should be quoted against a published Vicsek value
    without that conversion being done explicitly. What IS comparable is the
    qualitative structure -- a susceptibility peak that grows with N, and a
    Binder crossing -- which is what a substrate validation needs.
    """

    sizes: Tuple[int, ...] = (100, 200, 400, 800)
    #: Dense through the transition (pilot: chi peaks near eta = 11 at N = 200),
    #: sparse in the wings, which is what brackets a transition rather than
    #: sampling a ladder uniformly and resolving nothing.
    eta: Tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 9.0, 9.5, 10.0, 10.5,
                              11.0, 11.5, 12.0, 12.5, 13.0, 14.0, 16.0, 20.0)
    steps: int = 600
    #: Steps discarded before measuring. Half the run: the order parameter
    #: relaxes from a random initial condition and the relaxation is not the
    #: fluctuation. `drift` reports whether this was enough, per cell.
    burn_in: int = 300
    dt: float = 0.1
    seeds: int = 20
    baseline: str = "vicsek_alignment"

    def key(self) -> Dict[str, object]:
        return {**asdict(self), "suite_version": SUITE_VERSION}


def _psi(vel: np.ndarray) -> float:
    speed = np.linalg.norm(vel, axis=1, keepdims=True)
    return float(np.linalg.norm((vel / np.maximum(speed, 1e-9)).mean(axis=0)))


def run_cell(n: int, eta: float, seed: int, spec: CriticalitySpec,
             engine: Optional[PhysicsEngine] = None) -> Dict[str, object]:
    """One ``(N, eta, seed)`` cell. Returns the psi series' moments, not a verdict.

    The series is collected through ``run_benchmark``'s observer rather than by
    re-implementing the stepping loop, so the ladder and the leaderboard are
    provably the same experiment on the same substrate.
    """
    scenario = VicsekTransitionScenario(noise_eta=float(eta), num_agents=int(n))
    series: List[float] = []
    burn = int(spec.burn_in)

    def observe(step: int, pos: np.ndarray, vel: np.ndarray) -> None:
        if step >= burn:
            series.append(_psi(vel))

    result = run_benchmark(
        scenario, vicsek_alignment(),
        BenchmarkConfig(steps=spec.steps, dt=spec.dt, seed=seed),
        baseline_name=spec.baseline, engine=engine, observer=observe)

    a = np.asarray(series, dtype=float)
    if a.size < 4:
        raise ValueError(
            f"cell (N={n}, eta={eta}, seed={seed}) measured {a.size} samples; "
            f"burn_in={spec.burn_in} leaves nothing to take a moment over. A "
            f"variance of fewer than two samples is 0.0, which reads as a "
            f"perfectly ordered swarm.")
    half = a.size // 2
    return {
        "n_agents": int(n), "eta": float(eta), "seed": int(seed),
        "bound": float(scenario.required_config["bound"]),
        "psi_mean": float(a.mean()),
        # Within-run fluctuation: THE susceptibility, per its definition.
        "chi": float(susceptibility(a, n)),
        "u4": float(binder_cumulant(a)),
        # Has it equilibrated? A moment over a relaxing series measures the
        # relaxation; this is the number that says whether burn_in was enough.
        "drift": float(a[half:].mean() - a[:half].mean()),
        "n_samples": int(a.size),
        # The row's own provenance: a criticality claim on a different substrate
        # is a different claim (DOCS 5).
        "engine": result.engine,
        "suite_version": result.suite_version,
        "terminal_metric": float(result.metric),
    }


def run_ladder(spec: Optional[CriticalitySpec] = None,
               engine: Optional[PhysicsEngine] = None,
               progress: bool = False) -> List[Dict[str, object]]:
    """Every cell of the ladder, flat. One engine is built and shared, so the
    ladder cannot mix substrates size-to-size."""
    spec = spec or CriticalitySpec()
    if engine is None:
        from gossamer.benchmarks.harness import default_engine
        engine = default_engine()
    rows: List[Dict[str, object]] = []
    for n in spec.sizes:
        for eta in spec.eta:
            for seed in range(spec.seeds):
                rows.append(run_cell(n, eta, seed, spec, engine))
        if progress:
            print(f"  N={n}: {len(rows)} cells", flush=True)
    return rows


# --------------------------------------------------------------------------
# reduction
# --------------------------------------------------------------------------

def _by_size_eta(rows: Sequence[Dict[str, object]]):
    out: Dict[Tuple[int, float], List[Dict[str, object]]] = {}
    for r in rows:
        out.setdefault((r["n_agents"], r["eta"]), []).append(r)
    return out


def reduce_ladder(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Pool seeds per ``(N, eta)``.

    ``chi`` is the mean of the WITHIN-RUN susceptibilities; ``chi_between_seed``
    is ``N * Var`` of the per-seed mean psi. They are different quantities that
    peak in the same place and differ by an order of magnitude, and reporting one
    of them under the other's name is the easiest mistake in this file.
    """
    out = []
    for (n, eta), cells in sorted(_by_size_eta(rows).items()):
        psi = np.array([c["psi_mean"] for c in cells], dtype=float)
        out.append({
            "n_agents": n, "eta": eta, "n_seeds": len(cells),
            "psi": float(psi.mean()),
            "psi_sd": float(psi.std(ddof=1)) if psi.size > 1 else 0.0,
            "chi": float(np.mean([c["chi"] for c in cells])),
            "chi_between_seed": float(n * psi.var(ddof=1)) if psi.size > 1 else 0.0,
            "u4": float(np.mean([c["u4"] for c in cells])),
            "u4_sd": float(np.std([c["u4"] for c in cells], ddof=1))
            if len(cells) > 1 else 0.0,
            "max_abs_drift": float(np.max(np.abs([c["drift"] for c in cells]))),
        })
    return out


def susceptibility_peaks(reduced: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Where chi peaks for each size, and how tall the peak is.

    A peak that does NOT grow with N is the finite-size signature of a crossover,
    not a transition -- so the height is reported beside the location rather than
    the location alone.
    """
    peaks = []
    sizes = sorted({r["n_agents"] for r in reduced})
    for n in sizes:
        rows = sorted([r for r in reduced if r["n_agents"] == n],
                      key=lambda r: r["eta"])
        i = int(np.argmax([r["chi"] for r in rows]))
        peaks.append({"n_agents": n, "eta_peak": rows[i]["eta"],
                      "chi_peak": rows[i]["chi"],
                      "on_grid_edge": i in (0, len(rows) - 1)})
    return peaks


def binder_crossing(reduced: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Locate the Binder crossing for each consecutive size pair.

    Returns one entry per pair plus the spread across pairs. The spread is the
    load-bearing part: finite-size scaling says the curves cross at a single
    ``eta_c``, so pairs that disagree are evidence AGAINST a critical point. A
    single averaged crossing would hide exactly that.

    Linear interpolation on the sign change of ``U4(large) - U4(small)``. A pair
    with no sign change reports ``None`` -- not the nearest grid point, which
    would manufacture a crossing out of a grid that does not contain one.
    """
    sizes = sorted({r["n_agents"] for r in reduced})
    curves = {n: sorted([r for r in reduced if r["n_agents"] == n],
                        key=lambda r: r["eta"]) for n in sizes}
    pairs = []
    for small, large in zip(sizes, sizes[1:]):
        a, b = curves[small], curves[large]
        etas = [r["eta"] for r in a]
        assert [r["eta"] for r in b] == etas, "sizes measured on different grids"
        d = np.array([bb["u4"] - aa["u4"] for aa, bb in zip(a, b)], dtype=float)
        # Standard error on each difference, from the measured seed spread. A
        # crossing has to clear this as well as the absolute floor.
        sig = np.array([
            math.sqrt((aa.get("u4_sd", 0.0) ** 2 + bb.get("u4_sd", 0.0) ** 2)
                      / max(1, min(aa.get("n_seeds", 1), bb.get("n_seeds", 1))))
            for aa, bb in zip(a, b)], dtype=float)
        floor = np.maximum(SIGNIFICANT_U4, SIGNIFICANCE_SIGMAS * sig)
        # Two curves that never separate have not crossed -- they are the SAME
        # curve, which is the size-independence a crossover shows and the
        # strongest evidence against a critical point. Read naively the first
        # `d == 0` is a sign change, so a perfectly degenerate pair would report
        # a crossing at the leftmost grid point: guardrail 4, inside the
        # instrument that is supposed to detect it.
        separation = float(np.max(np.abs(d))) if d.size else 0.0
        coincident = separation < COINCIDENT_U4
        crossings: List[float] = []
        if not coincident:
            for i in range(len(d) - 1):
                if d[i] * d[i + 1] > 0.0:
                    continue                      # same side, no sign change
                # Significant on at least one side, or it is noise in a region
                # where the curves are indistinguishable.
                if max(abs(d[i]), abs(d[i + 1])) < max(floor[i], floor[i + 1]):
                    continue
                if d[i] == d[i + 1]:              # both zero: not a crossing
                    continue
                w = abs(d[i]) / (abs(d[i]) + abs(d[i + 1]))
                crossings.append(etas[i] + w * (etas[i + 1] - etas[i]))
        # A crossing that lands exactly ON a grid point has a zero difference
        # there, and the intervals on both sides of it each report it -- so one
        # crossing arrives as two and the pair reads as ambiguous. Merge
        # crossings closer together than half a grid step.
        if len(crossings) > 1 and len(etas) > 1:
            step = min(etas[i + 1] - etas[i] for i in range(len(etas) - 1))
            merged = [crossings[0]]
            for c in crossings[1:]:
                if abs(c - merged[-1]) > 0.5 * step:
                    merged.append(c)
            crossings = merged

        # `monotone` is the evidence AGAINST a crossing, and it is the finding
        # when there is no crossing: if the larger size is above the smaller at
        # every eta, the curves cannot have crossed inside the sweep.
        monotone = bool(np.all(d > 0.0)) or bool(np.all(d < 0.0))
        pairs.append({"sizes": [small, large],
                      "eta_cross": crossings[0] if len(crossings) == 1 else None,
                      "n_significant_crossings": len(crossings),
                      "all_crossings": crossings,
                      "max_u4_separation": separation,
                      "monotone_in_n": monotone,
                      "coincident": coincident})
    found = [p["eta_cross"] for p in pairs if p["eta_cross"] is not None]
    return {
        "pairs": pairs,
        "n_pairs_with_crossing": len(found),
        "n_pairs_coincident": sum(1 for p in pairs if p["coincident"]),
        "n_pairs_monotone": sum(1 for p in pairs if p["monotone_in_n"]),
        "n_pairs_ambiguous": sum(1 for p in pairs
                                 if p["n_significant_crossings"] > 1),
        "eta_c_mean": float(np.mean(found)) if found else None,
        "eta_c_spread": float(np.max(found) - np.min(found)) if len(found) > 1 else None,
    }


#: A cell whose psi still moves this much across the measured window has not
#: equilibrated, and its moments describe the relaxation.
DRIFT_TOLERANCE = 0.05

#: Two Binder curves separated by less than this over the whole sweep are
#: treated as the same curve. Size-independence is what a crossover looks like,
#: and it is NOT a crossing however the sign of the difference wanders.
COINCIDENT_U4 = 1e-6

#: A sign change is only a crossing if the curves are meaningfully apart on at
#: least one side of it. THIS WAS WRONG FIRST and the wrong version produced a
#: confident number: in the ordered phase U4 is pinned at 2/3 to five decimals
#: for every size, the difference wanders at the 1e-5 level, and taking the
#: first sign change from the left reported eta_c = 0.0018 for a transition the
#: susceptibility puts near eta = 12. A crossing found where the curves are flat
#: is an artefact of scanning order, not a measurement.
SIGNIFICANT_U4 = 1e-3

#: How many standard errors of separation a sign change needs, on top of the
#: absolute floor. The two together mean a crossing must be both physically
#: meaningful and larger than the seed noise.
SIGNIFICANCE_SIGMAS = 3.0


def audit_ladder(rows: Sequence[Dict[str, object]],
                 reduced: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Mechanical checks. ``fail`` means a number here cannot be reported as-is."""
    rep: Dict[str, object] = {}

    # The ladder must BRACKET the transition, or it has measured one phase and
    # the peak it reports is the edge of the grid (guardrail 4).
    peaks = susceptibility_peaks(reduced)
    rep["susceptibility_peaks"] = peaks
    rep["peak_on_grid_edge"] = [p for p in peaks if p["on_grid_edge"]]

    ordered = [r for r in reduced if r["psi"] > 0.9]
    disordered = [r for r in reduced if r["psi"] < 0.2]
    rep["brackets_transition"] = bool(ordered and disordered)

    # A susceptibility peak that does not grow with N is a crossover signature.
    heights = [p["chi_peak"] for p in sorted(peaks, key=lambda p: p["n_agents"])]
    rep["chi_peak_grows_with_n"] = bool(
        len(heights) > 1 and heights[-1] > heights[0])
    rep["chi_peak_heights"] = heights

    # Equilibration.
    bad = [r for r in reduced if r["max_abs_drift"] > DRIFT_TOLERANCE]
    rep["cells_not_equilibrated"] = [
        {"n_agents": r["n_agents"], "eta": r["eta"],
         "max_abs_drift": r["max_abs_drift"]} for r in bad]

    # The two susceptibilities, so nobody reports one as the other.
    ratios = [r["chi_between_seed"] / r["chi"] for r in reduced if r["chi"] > 0]
    rep["between_over_within_chi"] = {
        "median": float(np.median(ratios)) if ratios else None,
        "max": float(np.max(ratios)) if ratios else None}

    rep["binder"] = binder_crossing(reduced)

    # One substrate, one suite version.
    rep["engines"] = sorted({r["engine"] for r in rows})
    rep["suite_versions"] = sorted({r["suite_version"] for r in rows})

    rep["PASS"] = bool(rep["brackets_transition"]
                       and not rep["peak_on_grid_edge"]
                       and not rep["cells_not_equilibrated"]
                       and len(rep["engines"]) == 1
                       and len(rep["suite_versions"]) == 1)
    return rep
