"""
Pure-numpy peer-state predictors — the P3 baselines (and gate).

* :class:`ConstantVelocityPredictor` — ``pos + vel · horizon·dt``. Exact on any
  constant-velocity trajectory; the minimal anticipatory result.
* :class:`KalmanPredictor` — a per-agent constant-velocity Kalman filter. Adds
  a covariance so calibration can report NIS (filter consistency).
* :class:`LinearPredictor` — least-squares line fit over the history window,
  extrapolated. Robust to mild curvature without a motion model.

All are deterministic given their inputs; none depend on torch (the learned
graph predictor lives separately, mirroring how ``gossamer.learning`` isolates
torch).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from gossamer.prediction.base import STATE_DIM, PeerHistory, PeerPredictor


class ConstantVelocityPredictor(PeerPredictor):
    """Extrapolate at the last-known velocity: ``x' = x + v · h·dt``."""

    name = "const_vel"

    def predict(self, history: PeerHistory, horizon_steps: int, dt: float) -> np.ndarray:
        latest = history.latest().copy()
        h = float(horizon_steps) * float(dt)
        latest[:, :3] = latest[:, :3] + latest[:, 3:6] * h
        return latest


class LinearPredictor(PeerPredictor):
    """Least-squares linear fit of position over the window, extrapolated.

    Velocity is taken from the fitted slope. Falls back to constant-velocity
    when fewer than two snapshots are available.
    """

    name = "linear"

    def __init__(self, window: int = 4):
        self.window = max(2, int(window))

    def predict(self, history: PeerHistory, horizon_steps: int, dt: float) -> np.ndarray:
        w = history.window(self.window)  # (w, N, 6)
        if w.shape[0] < 2:
            return ConstantVelocityPredictor().predict(history, horizon_steps, dt)
        wlen = w.shape[0]
        t = np.arange(wlen, dtype=float)              # (w,)
        t_future = float(wlen - 1 + horizon_steps)
        # Fit x ≈ a·t + b per agent per axis via closed-form least squares.
        tbar = t.mean()
        denom = float(((t - tbar) ** 2).sum()) or 1.0
        pos = w[:, :, :3]                              # (w, N, 3)
        pbar = pos.mean(axis=0)                        # (N, 3)
        slope = np.einsum("t,tna->na", (t - tbar), pos - pbar[None]) / denom  # (N,3)
        intercept = pbar - slope * tbar
        out = history.latest().copy()
        out[:, :3] = slope * t_future + intercept
        out[:, 3:6] = slope / max(dt, 1e-9)
        return out


class KalmanPredictor(PeerPredictor):
    """Per-agent constant-velocity Kalman filter (position + velocity).

    Each agent's state is ``[x, v]`` per axis, tracked independently. On each
    ``predict`` the filter ingests every snapshot in the history it has not yet
    seen (predict→update), then rolls the posterior forward ``horizon_steps``.
    Exposes the predicted position covariance so :meth:`calibration` can report
    NIS for chi-square consistency checks.
    """

    name = "kalman"

    def __init__(self, process_var: float = 1e-2, meas_var: float = 1.0):
        self.q = float(process_var)
        self.r = float(meas_var)
        self._n: int = 0
        self._x: Optional[np.ndarray] = None   # (N, 3, 2)  [pos, vel] per axis
        self._P: Optional[np.ndarray] = None   # (N, 3, 2, 2)
        self._seen: int = 0
        self._last_pos_cov: Optional[np.ndarray] = None

    def reset(self, num_agents: int) -> None:
        self._n = int(num_agents)
        self._x = None
        self._P = None
        self._seen = 0
        self._last_pos_cov = None

    def _ensure_init(self, first: np.ndarray) -> None:
        n = first.shape[0]
        self._n = n
        self._x = np.zeros((n, 3, 2))
        self._x[:, :, 0] = first[:, :3]
        self._x[:, :, 1] = first[:, 3:6]
        self._P = np.tile(np.eye(2) * 10.0, (n, 3, 1, 1))
        self._seen = 1

    def _step_filter(self, meas: np.ndarray, dt: float) -> None:
        """One predict→update over a position measurement ``meas`` (N,6)."""
        n = self._n
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        H = np.array([[1.0, 0.0]])
        # Predict: x = F x ; P = F P F^T + Q  (broadcast over N×3 independent filters)
        x = np.einsum("ij,naj->nai", F, self._x)                       # (N,3,2)
        P = np.einsum("ij,nakj->naik", F, self._P)
        P = np.einsum("naik,lk->nail", P, F) + Q[None, None]           # (N,3,2,2)
        # Update with position measurement.
        z = meas[:, :3]                                                # (N,3)
        y = z - x[:, :, 0]                                             # innovation (N,3)
        S = P[:, :, 0, 0] + self.r                                     # (N,3)
        K = P[:, :, :, 0] / S[:, :, None]                             # (N,3,2)
        self._x = x + K * y[:, :, None]
        I_KH = np.eye(2)[None, None] - np.einsum("nai,j->naij", K, H[0])
        self._P = np.einsum("naij,najk->naik", I_KH, P)

    def predict(self, history: PeerHistory, horizon_steps: int, dt: float) -> np.ndarray:
        win = history.window(history.capacity)  # oldest→newest
        if win.shape[0] == 0:
            raise ValueError("empty history")
        if self._x is None:
            self._ensure_init(win[0])
        # Ingest any snapshots newer than what we've filtered.
        for k in range(self._seen, win.shape[0]):
            self._step_filter(win[k], dt)
            self._seen = k + 1
        # Roll the posterior forward `horizon_steps` without new measurements.
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = self.q * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        x, P = self._x.copy(), self._P.copy()
        for _ in range(int(horizon_steps)):
            x = np.einsum("ij,naj->nai", F, x)
            P = np.einsum("ij,nakj->naik", F, P)
            P = np.einsum("naik,lk->nail", P, F) + Q[None, None]
        self._last_pos_cov = P[:, :, 0, 0]  # (N,3) predicted position variance
        out = history.latest().copy()
        out[:, :3] = x[:, :, 0]
        out[:, 3:6] = x[:, :, 1]
        return out

    def calibration(self, predicted: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        base = super().calibration(predicted, actual)
        if self._last_pos_cov is not None:
            err = np.asarray(predicted)[:, :3] - np.asarray(actual)[:, :3]
            var = np.maximum(self._last_pos_cov, 1e-12)
            nis = float(np.mean(np.sum(err ** 2 / var, axis=1)))  # ~χ²(3) when consistent
            base["nis"] = nis
        return base


PREDICTORS = {
    "const_vel": ConstantVelocityPredictor,
    "linear": LinearPredictor,
    "kalman": KalmanPredictor,
}


__all__ = [
    "ConstantVelocityPredictor",
    "KalmanPredictor",
    "LinearPredictor",
    "PREDICTORS",
]
