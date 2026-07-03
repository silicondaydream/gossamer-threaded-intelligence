"""
Peer-state prediction interface for anticipatory coordination (P3).

Under communication delay, an agent's view of its peers is stale by the light-
lag. A :class:`PeerPredictor` extrapolates peers' *current* state from their
last-known (delayed) states, so the coordination primitive can decide on the
predicted state and later reconcile against the CRDT ground truth when the real
delayed message arrives (the "decide-on-prediction, reconcile-with-CRDT" loop).

The minimal, gating result for P3 is the pure-kinematic baselines in
:mod:`gossamer.prediction.baselines` (constant-velocity / Kalman / linear); the
learned graph predictor is the stretch, not the gate.

State convention: a peer *state* row is ``[x, y, z, vx, vy, vz]`` (position then
velocity). ``predict`` maps a history of such rows to the estimated state
``horizon_steps`` into the future; ``calibration`` scores a batch of predictions
against realized states.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

STATE_DIM = 6  # [x, y, z, vx, vy, vz]


class PeerHistory:
    """Fixed-length ring buffer of the most recent (delayed) peer states.

    ``push`` appends a ``(N, 6)`` snapshot; ``window`` returns the last ``w``
    snapshots as ``(w, N, 6)`` (oldest first). Deterministic and allocation-light
    so it can run inside the per-step loop at scale.
    """

    def __init__(self, num_agents: int, capacity: int = 8):
        self.num_agents = int(num_agents)
        self.capacity = max(1, int(capacity))
        self._buf: list = []

    def push(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=float)
        if state.shape != (self.num_agents, STATE_DIM):
            raise ValueError(f"state must be ({self.num_agents}, {STATE_DIM}), got {state.shape}")
        self._buf.append(state.copy())
        if len(self._buf) > self.capacity:
            self._buf.pop(0)

    def __len__(self) -> int:
        return len(self._buf)

    def latest(self) -> np.ndarray:
        return self._buf[-1]

    def window(self, w: int) -> np.ndarray:
        w = min(int(w), len(self._buf))
        return np.stack(self._buf[-w:], axis=0) if w > 0 else np.empty((0, self.num_agents, STATE_DIM))


class PeerPredictor(ABC):
    """Extrapolate current peer state from delayed history."""

    name: str = "base"

    def reset(self, num_agents: int) -> None:
        """Initialise per-run internal state. Default: stateless (no-op)."""

    @abstractmethod
    def predict(self, history: PeerHistory, horizon_steps: int, dt: float) -> np.ndarray:
        """Return predicted ``(N, 6)`` state ``horizon_steps`` ahead of the
        latest snapshot in ``history``. ``horizon_steps == 0`` must return the
        latest snapshot unchanged (the zero-delay identity)."""
        ...

    def calibration(self, predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Score predictions against realized states.

        Returns position RMSE and (when the predictor exposes a covariance) the
        normalized innovation squared (NIS) for filter consistency. The base
        implementation reports RMSE only.
        """
        predicted = np.asarray(predicted, dtype=float)
        actual = np.asarray(actual, dtype=float)
        pos_err = predicted[:, :3] - actual[:, :3]
        rmse = float(np.sqrt(np.mean(np.sum(pos_err ** 2, axis=1)))) if pos_err.size else 0.0
        return {"pos_rmse": rmse}


__all__ = ["PeerHistory", "PeerPredictor", "STATE_DIM"]
