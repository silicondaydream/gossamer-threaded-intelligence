"""What an agent BELIEVES about its peers — stale, gated, predicted, and noisy.

This is the mechanism the whole DCC corpus is built on, so it is worth stating
plainly: the coordination primitives must decide on **stale** peer state, or Q is
independent of the delay axis and the phase diagram is measuring nothing. The
engine still integrates the TRUE state from the acceleration the caller hands
back; only the *view* is delayed.

**Why it lives in Gossamer.** It used to live in the Maneuver.Map runner, which is
the orchestration layer — the same misplacement the original Gossamer extraction
made with `policies.py`, and costly for the same three reasons (DOCS §2): an
algorithm in the runner is not covered by Gossamer's tests, not shipped in the
wheel the papers pin, and **not runnable by anyone reproducing a DCC result from
the public package**. It is also what blocked the DCC delay task from becoming a
benchmark row: `orrery.benchmarks` cannot import the runner, so the crown-jewel
task of the constellation benchmark had no substrate. A reader would call this
*the method*, so it belongs here.

Four transformations, applied in this order, and the order is the physics:

1. **Delay** (Gap A, the core P1 mechanism). A ring of the last `delay_steps + 1`
   true (pos, vel) frames; the primitive at step i reads the snapshot from
   `i - delay_steps`. Early in the run the ring is short, so it reads the oldest it
   has — which is correct: information that was never sent cannot arrive.

2. **Delivery gating** (P5). When gating is on, the caller reports which ordered
   links actually carried a bundle, and the primitive's interaction graph is
   intersected with them — so bandwidth / loss / range act *causally* on
   coordination quality. Without it the primitives read the full delayed view while
   the comm model is a passive cost meter, which is why the P2 cost frontier was
   degenerate: 0 bits/s still gave Q = 1.0.

   The alignment is subtle and load-bearing. Leviathan's `AgentManager::step` runs
   physics THEN the channel, so the edges the engine reports were computed on
   exactly the (pos, vel) that same step returned. Passing them in at the top of the
   next iteration pairs each frame with its own edge set, and the ring carries the
   pair together — so an edge recorded at transmit time surfaces `delay_steps`
   later, which is when it would actually have arrived.

   **The edges are passed IN, not pulled from an engine.** That inversion is what
   makes this module importable by a benchmark: the caller owns the engine, this
   owns the mechanism, and the seam between them is an array (DOCS §2, "cross-repo
   seams pass arrays, not imports"). The capability check that used to live in this
   constructor — does the engine actually expose `comm_edges`? — moved to the
   caller, which is the only side that can answer it; `perceive` refuses a `None`
   edge set while gating rather than silently coordinating ungated.

3. **Prediction** (P3). The predictor extrapolates the stale view forward to estimate
   where the peers are *now*, and the primitive coordinates on that instead — this is
   how delay-lost Q is recovered. The extrapolation is scored against the realized
   ground truth for calibration, and this is the ONLY place that holds both, which is
   why the calibration is written here rather than in an observer's `observe()`.
   Prediction never mutates ground truth; it only fills the delay gap for the decision.

4. **Sensing noise** (DMB §4.2). The *observation* channel, so it is policy-side: it
   perturbs the perceived peer-position field, never the true state the engine
   integrates — which is why it cannot inject energy the way a velocity perturbation
   would. Positions only; velocities are left true so that the caller's true-velocity
   re-actuation stays energy-neutral. Drawn from the run's seeded `rng`, held by
   reference, so the draw order is exactly what it was inline.

A misconfigured predictor ABORTS. It used to be `except Exception: _predictor = None`,
which silently downgraded a P3 "kalman" cell to the no-prediction baseline — a
different experimental condition, reported under the label of the one that was asked
for. `model == "none"` remains a legitimate way to ask for no predictor.
"""
from __future__ import annotations

import collections
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["DelayedView", "resolve_delay_steps"]

#: A ring of `delay_steps + 1` float32 frames is the memory cost of the delay axis, so
#: it is capped. Override with DCC_MAX_DELAY_STEPS.
_DEFAULT_MAX_DELAY_STEPS = 2000


def resolve_delay_steps(config_map: Dict[str, str], dt: float) -> int:
    """Comm latency (ms) -> steps, falling back to an explicit step count.

    `comm_latency_ms` and `coupling_delay_steps` are two spellings of one axis; the
    engine's typed config already refuses to let `comm_latency_ms` and
    `comm_latency_steps` shadow each other (they used to, and ms silently won).
    """
    lat_ms = float(config_map.get("comm_latency_ms", 0.0) or 0.0)
    if lat_ms > 0 and dt > 0:
        steps = int(round((lat_ms / 1000.0) / dt))
    else:
        steps = int(config_map.get("coupling_delay_steps", 0) or 0)
    steps = max(0, steps)
    try:
        max_delay = int(os.environ.get("DCC_MAX_DELAY_STEPS", str(_DEFAULT_MAX_DELAY_STEPS)))
    except Exception:
        max_delay = _DEFAULT_MAX_DELAY_STEPS
    return min(steps, max_delay)


def gating_enabled(config_map: Dict[str, str]) -> bool:
    """Whether delivery gating is on, by the one spelling of the truth test.

    Exposed because the caller has to make the same decision — it must know whether
    to fetch delivered edges from its engine — and two copies of this predicate is
    how gating silently ends up on for the view and off for the fetch.
    """
    return str(config_map.get("comm_collect_edges", "0")).strip().lower() \
        not in ("", "0", "false", "no")


class DelayedView:
    """The perceived peer state, one step at a time.

    Holds `rng` BY REFERENCE — the sensing-noise draws share the run's single seeded
    stream, and moving them onto a stream of their own would renumber every draw after
    them and move published numbers.
    """

    def __init__(self, config_map: Dict[str, str], dt: float, num_agents: int,
                 prediction: Optional[Dict[str, Any]],
                 rng: np.random.Generator,
                 delay_steps_per_agent: Optional[np.ndarray] = None) -> None:
        self.dt = float(dt)
        self._rng = rng

        # --- the delay axis: one scalar knob, or one offset per agent ---
        #
        # `delay_steps_per_agent` is what turns the delay axis from DIALLED into
        # MEASURED: the offsets come from `orrery.astro.delay` over a real contact
        # plan, so a peer's staleness is its own contact geometry rather than a
        # number typed into a preset. Gossamer never imports Orrery — the array
        # arrives across the seam (DOCS §2), which is also why this takes steps and
        # not a plan.
        self.per_agent_delay: Optional[np.ndarray] = None
        if delay_steps_per_agent is None:
            self.delay_steps = resolve_delay_steps(config_map, dt)
            maxlen = self.delay_steps + 1
        else:
            arr = np.asarray(delay_steps_per_agent)
            if arr.ndim != 1 or arr.shape[0] != num_agents:
                raise ValueError(
                    f"delay_steps_per_agent must be one offset per agent, got shape "
                    f"{arr.shape} for num_agents={num_agents}.")
            if not np.issubdtype(arr.dtype, np.integer):
                # A float offset would be silently truncated into a different
                # experiment. The seconds->steps rounding belongs upstream, where the
                # oracle can refuse an unreachable pair instead of flattening it.
                raise TypeError(
                    "delay_steps_per_agent must be integer steps; convert seconds "
                    "upstream with orrery.astro.delay.to_engine_delay_steps, which "
                    "refuses a non-finite (unreachable) delay rather than rounding it.")
            if np.any(arr < 0):
                raise ValueError("delay_steps_per_agent must be non-negative.")
            self.per_agent_delay = arr.astype(int, copy=True)
            # The scalar stays defined as the deepest offset: it is what the ring has
            # to hold, and what `delay_steps > 0` means for the predictor gate.
            self.delay_steps = int(self.per_agent_delay.max(initial=0))
            maxlen = self.delay_steps + 1

        # Oldest frame in the ring IS the delayed view (scalar case). `edges` is None
        # unless gating.
        self._ring: collections.deque = collections.deque(maxlen=maxlen)

        self.gated = gating_enabled(config_map)

        self.sensing_noise = float(config_map.get("sensing_noise", 0.0) or 0.0)

        # --- the predictor (P3) ---
        self.predictor = None
        self._peer_hist = None
        self.pred_delay_steps = self.delay_steps
        #: Written HERE, read by Maneuver.Map's `metrics_registry.PredictionObserver`.
        #: This is the only place holding both the prediction and the truth it is
        #: scored against.
        self.calibration: Dict[str, List[float]] = {"pos_rmse": [], "nis": []}
        self.model = str((prediction or {}).get("model", "const_vel")).lower()
        if prediction:
            from gossamer.prediction.baselines import PREDICTORS as _PREDICTORS
            from gossamer.prediction.base import PeerHistory as _PeerHistory
            model = str(prediction.get("model", "const_vel")).lower()
            if model != "none":
                pcls = _PREDICTORS.get(model)
                if pcls is None:
                    raise ValueError(
                        f"unknown predictor {model!r}; expected 'none' or one of "
                        f"{sorted(_PREDICTORS)}")
                if self.per_agent_delay is not None:
                    # `predict()` takes ONE horizon, so combining it with per-agent
                    # offsets would extrapolate every peer by the deepest one — a
                    # different experiment for all but the slowest agent, and silent.
                    # Refusing is the honest state until the predictors take a
                    # per-agent horizon; this is not a fallback worth guessing at.
                    raise ValueError(
                        "per-agent delay offsets and a predictor are not yet "
                        "combinable: the predictors extrapolate by a single scalar "
                        "horizon, so every peer would be projected forward by the "
                        "deepest offset. Use prediction={'model': 'none'}, or a "
                        "scalar delay.")
                self.predictor = pcls(**(prediction.get("params") or {}))
                self.predictor.reset(num_agents)
                self._peer_hist = _PeerHistory(
                    num_agents, capacity=int(prediction.get("history", 8)))
                # A predictor at zero delay would be asked to extrapolate zero steps.
                self.pred_delay_steps = max(1, self.delay_steps)

    def _gather_per_agent(self) -> Tuple[np.ndarray, np.ndarray, Any]:
        """One row per agent, each read from the frame ITS OWN offset steps back.

        Ring index `L - 1 - d` is the frame `d` steps old (`L - 1` is this step's).
        Clamping at 0 reproduces the scalar path's "read the oldest you have" while
        the ring is still filling — information that was never sent cannot arrive.

        **This reduces EXACTLY to the scalar path when every offset is equal**, which
        is the property that lets the measured-delay work land without re-pinning a
        published number: at a uniform `d`, `L - 1 - d` is 0 once the ring is full and
        clamps to 0 before that — the same frame `self._ring[0]` returns. Locked by
        `test_a_uniform_per_agent_offset_is_the_scalar_path`.
        """
        L = len(self._ring)
        idx = np.maximum(0, L - 1 - self.per_agent_delay)  # (N,) ring positions

        newest_pos = self._ring[-1][0]
        view_pos = np.empty(newest_pos.shape, dtype=np.float64)
        view_vel = np.empty(newest_pos.shape, dtype=np.float64)
        edge_blocks = []
        for k in np.unique(idx):
            frame_pos, frame_vel, frame_edges = self._ring[int(k)]
            sel = idx == k
            view_pos[sel] = frame_pos[sel].astype(np.float64)
            view_vel[sel] = frame_vel[sel].astype(np.float64)
            if self.gated and frame_edges is not None and len(frame_edges) > 0:
                # An edge (src -> dst) surfaces when it ARRIVES, and its flight time
                # is the SOURCE's offset: dst's picture of src is `d_src` stale, so
                # the link that carried it is too. Keep the rows of this frame's edge
                # list whose src sits at this offset.
                e = np.asarray(frame_edges)
                edge_blocks.append(e[sel[e[:, 0]]])
        if not self.gated:
            return view_pos, view_vel, None
        dv_edges = (np.concatenate(edge_blocks, axis=0) if edge_blocks
                    else np.zeros((0, 2), dtype=int))
        return view_pos, view_vel, dv_edges

    def perceive(self, pos: np.ndarray, vel: np.ndarray,
                 edges_now: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Any]:
        """Push this step's truth into the ring; return what the primitive gets to see.

        `edges_now` is the set of ordered links that carried a bundle THIS step, as
        the caller's engine reported it — required when gating is on, ignored when it
        is off. Returns `(view_pos, view_vel, edges)`, where `edges` is the delivered-
        link set that travelled with the frame being surfaced, so it is
        `delay_steps` old too.
        """
        # Gating on with no edges would coordinate on the full delayed view while
        # reporting itself as gated — a different experimental condition under the
        # asked-for one's label, and silent. The caller's engine check happens once
        # at construction; this catches the per-step half of the same mistake.
        if self.gated and edges_now is None:
            raise ValueError(
                "delivery gating is enabled (comm_collect_edges) but perceive() got no "
                "edges. Pass the delivered-link array from the engine, or turn gating "
                "off — coordinating ungated under a gated label is not a fallback.")
        self._ring.append((pos.astype(np.float32, copy=True),
                           vel.astype(np.float32, copy=True),
                           edges_now if self.gated else None))
        if self.per_agent_delay is None:
            dv_pos32, dv_vel32, dv_edges = self._ring[0]
            view_pos = dv_pos32.astype(np.float64)
            view_vel = dv_vel32.astype(np.float64)
        else:
            view_pos, view_vel, dv_edges = self._gather_per_agent()

        # Decide-on-prediction: extrapolate the stale view back to "now".
        if self.predictor is not None and pos.size and self.delay_steps > 0:
            self._peer_hist.push(np.concatenate([view_pos, view_vel], axis=1))
            if len(self._peer_hist) >= 2:
                pred = self.predictor.predict(self._peer_hist, self.pred_delay_steps, self.dt)
                if pred is not None and pred.shape[0] == pos.shape[0]:
                    # Score the extrapolation against the realized state NOW. No
                    # `except: pass` here: swallowing a calibration failure yields
                    # `pos_rmse_mean = 0.0` from an empty sample list — a PERFECT
                    # predictor, which is the P3 paper's headline column.
                    truth = np.concatenate([pos, vel], axis=1)
                    cal = self.predictor.calibration(pred, truth)
                    self.calibration["pos_rmse"].append(float(cal.get("pos_rmse", 0.0)))
                    if "nis" in cal:
                        self.calibration["nis"].append(float(cal["nis"]))
                    view_pos = pred[:, :3]
                    if pred.shape[1] >= 6:
                        view_vel = pred[:, 3:6]

        # Sensing noise on the PERCEIVED position field only (see the module docstring).
        if self.sensing_noise > 0.0 and view_pos.size:
            view_pos = view_pos + self._rng.normal(
                0.0, self.sensing_noise, size=view_pos.shape)

        return view_pos, view_vel, dv_edges
