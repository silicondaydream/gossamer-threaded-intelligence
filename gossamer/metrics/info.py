"""
Information-theoretic measures for agent-agent influence.

Why these matter: cohesion/alignment/separation only tell you the *state*
of the swarm. Mutual information and transfer entropy let you attribute
*causation* — "does what agent i did last step predict what agent j does
next step beyond what j's own history predicts?" That's the right
instrument for studying emergent leadership, influence backbones, and
information flow during phase transitions.

The estimators here are for continuous-valued scalar or low-dimensional
signals (e.g., per-agent speed, heading angle, SOC). For high-dimensional
signals you'll want to project first or use a neural estimator.

Estimator choice:

* Histogram (:func:`mutual_information_histogram`) — fast, unbiased at
  small dimensionality, but sensitive to bin count. Good default for 1D.
* Kraskov-Grassberger-Stögbauer (:func:`mutual_information_ksg`) — bias-
  corrected kNN estimator. Slower, but the standard in the information-
  theory literature. Use for papers.

Transfer entropy uses the Kraskov decomposition expressed in terms of
four mutual-information terms (the "KSG MI style" form).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.ndim != 2:
        raise ValueError(f"expected 1D or 2D array, got shape {x.shape}")
    return x


# ---- Histogram MI (fast baseline) ----

def mutual_information_histogram(
    x: np.ndarray,
    y: np.ndarray,
    bins: int = 32,
) -> float:
    """Discrete-bin MI estimate in nats.

    Both signals are 1D. For multidimensional signals use
    :func:`mutual_information_ksg`.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape[0] != y.shape[0] or x.size < 2:
        return 0.0
    # 2D histogram
    hxy, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hxy / max(hxy.sum(), 1.0)
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(pxy > 0, np.log(pxy / (px * py + 1e-300) + 1e-300), 0.0)
    return float(np.sum(pxy * log_term))


# ---- Kraskov-Grassberger-Stögbauer MI (paper-grade) ----

def _pairwise_chebyshev(a: np.ndarray) -> np.ndarray:
    """All-pairs Chebyshev (L-inf) distance for small N."""
    # a: (N, D)
    diff = np.abs(a[:, None, :] - a[None, :, :])
    return diff.max(axis=2)


def mutual_information_ksg(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 4,
) -> float:
    """KSG (Kraskov-Grassberger-Stögbauer, 2004) MI estimator in nats.

    Implements estimator form 1: for each pair ``(x_i, y_i)`` find the
    distance to the k-th nearest neighbor in the joint ``(x, y)`` space
    under the Chebyshev norm, then count how many points lie within that
    distance in each marginal.

    ``O(N^2)`` in N; fine up to a few thousand samples. For larger N use
    a KDTree-backed variant (``scipy.spatial.cKDTree``) which we fall
    back to when scipy is available.
    """
    X = _ensure_2d(x)
    Y = _ensure_2d(y)
    if X.shape[0] != Y.shape[0]:
        raise ValueError("x and y must have the same number of samples")
    n = X.shape[0]
    if n <= k + 1:
        return 0.0

    try:
        from scipy.spatial import cKDTree  # type: ignore
        from scipy.special import digamma  # type: ignore
    except Exception:  # pragma: no cover - fall back to O(N^2)
        cKDTree = None

        def digamma(z):  # type: ignore[misc]
            # Good-enough approximation for n >= 2
            z = np.asarray(z, dtype=float)
            return np.log(z) - 0.5 / z

    if cKDTree is not None:
        joint = np.concatenate([X, Y], axis=1)
        tree_xy = cKDTree(joint)
        # k+1 because the nearest neighbor includes the point itself
        d, _ = tree_xy.query(joint, k=k + 1, p=np.inf)
        eps = d[:, -1]  # distance to k-th neighbor (excluding self)
        tree_x = cKDTree(X)
        tree_y = cKDTree(Y)
        # Count neighbors strictly closer than eps in each marginal
        nx = np.array([len(tree_x.query_ball_point(X[i], r=eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
        ny = np.array([len(tree_y.query_ball_point(Y[i], r=eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)])
    else:
        joint = np.concatenate([X, Y], axis=1)
        d_xy = _pairwise_chebyshev(joint)
        np.fill_diagonal(d_xy, np.inf)
        eps = np.sort(d_xy, axis=1)[:, k - 1]
        d_x = _pairwise_chebyshev(X)
        np.fill_diagonal(d_x, np.inf)
        d_y = _pairwise_chebyshev(Y)
        np.fill_diagonal(d_y, np.inf)
        nx = (d_x < eps[:, None]).sum(axis=1)
        ny = (d_y < eps[:, None]).sum(axis=1)

    # KSG estimator form 1 in nats
    return float(
        digamma(np.asarray(k, dtype=float))
        + digamma(np.asarray(n, dtype=float))
        - np.mean(digamma(nx + 1.0) + digamma(ny + 1.0))
    )


# ---- Transfer entropy ----

def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    k: int = 4,
    history: int = 1,
) -> float:
    """Transfer entropy TE(source -> target) in nats.

    TE = I(target_{t+lag}; source_t | target_t^{history}), i.e. how much
    knowing the source's current value reduces uncertainty in the target's
    future beyond what the target's own past already supplied.

    Computed via the KSG decomposition::

        TE = I(Y_future; [S_t, Y_past]) - I(Y_future; Y_past)

    Small ``k`` and ``history`` are fine for preliminary research;
    published numbers should sweep both.
    """
    s = np.asarray(source, dtype=float).ravel()
    t = np.asarray(target, dtype=float).ravel()
    if s.shape[0] != t.shape[0] or s.size <= lag + history:
        return 0.0
    y_future = t[history + lag - 1 + 1:]  # shift by (history+lag-1)+1 so indices align
    # Build target past as a (n, history) window
    y_past_windows = np.stack(
        [t[history - 1 - j: t.size - lag - j] for j in range(history)],
        axis=1,
    )
    s_now = s[history - 1: s.size - lag].reshape(-1, 1)
    # Truncate to same n (indexing drift above can leave off-by-one)
    n = min(y_future.shape[0], y_past_windows.shape[0], s_now.shape[0])
    y_future = y_future[:n].reshape(-1, 1)
    y_past_windows = y_past_windows[:n]
    s_now = s_now[:n]

    joint = np.concatenate([s_now, y_past_windows], axis=1)
    i_full = mutual_information_ksg(y_future, joint, k=k)
    i_reduced = mutual_information_ksg(y_future, y_past_windows, k=k)
    return float(max(i_full - i_reduced, 0.0))


__all__ = [
    "mutual_information_histogram",
    "mutual_information_ksg",
    "transfer_entropy",
]
