"""
Learned multi-agent policies.

This package layers on top of :mod:`gossamer.graph` so the classical
message-passing abstraction and learned GNN policies share one surface.
PyTorch is a hard dependency *only* for symbols under this package; the
rest of Gossamer stays NumPy-only.

Modules:

* :mod:`gossamer.learning.comm_channel` — learnable message channel with
  configurable bandwidth, latency, and loss. Used by emergent-comm
  experiments.
* :mod:`gossamer.learning.policy_base` — CTDE shared-parameter GNN policy
  base class + value head; subclasses plug in message/update heads.
* :mod:`gossamer.learning.mappo` — reference MAPPO driver. Thin; expect
  to swap in CleanRL / MARLlib for production training.

Importing this package without PyTorch installed raises ``ImportError``
with a clear hint, rather than silently breaking at use time.
"""
from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as e:  # pragma: no cover - runtime environment check
    raise ImportError(
        "gossamer.learning requires PyTorch. Install it with 'pip install torch' "
        "or skip this subpackage; the rest of gossamer works with NumPy alone."
    ) from e

from gossamer.learning.comm_channel import CommChannel, CommChannelConfig
from gossamer.learning.policy_base import GraphActorCritic, SharedMLPHead

__all__ = [
    "CommChannel",
    "CommChannelConfig",
    "GraphActorCritic",
    "SharedMLPHead",
]
