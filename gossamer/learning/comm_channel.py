"""
Learnable communication channel with realistic physical-layer constraints.

Emergent-communication research wants a channel that's: (a) differentiable
end-to-end so messages can be trained, and (b) physically plausible so
the protocols that emerge are not trivially exploiting arbitrary
bandwidth. This channel implements:

* Bandwidth as a hard per-message dimensionality cap with a straight-through
  quantizer when ``n_levels`` is set (discretized codes a la EGG / DIAL).
* Latency as a ring-buffer delay in simulation steps.
* Loss as a Bernoulli mask per edge per step with gradient pass-through.
* Energy accounting proportional to ``bits * j_per_kb`` so energy-aware
  bids have a real underlying cost to reason about.

All of this is implemented in PyTorch so it composes with the GNN
policies in :mod:`gossamer.learning.policy_base` and trains with standard
MARL losses.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class CommChannelConfig:
    """Physical-layer constraints on the learned channel.

    ``message_dim`` — size of the continuous message vector carried per
    edge each step. Tune to match your bandwidth budget when paired with
    ``n_levels``.
    ``n_levels`` — if set, quantize each dimension to this many levels
    (1 bit ≈ 2 levels). ``None`` disables quantization.
    ``latency_steps`` — integer simulation-step delay before a message
    arrives. 0 means "delivered in the same step."
    ``loss_prob`` — probability of dropping an individual message per
    step.
    ``j_per_kb`` — energy cost per kilobit transmitted. Informational
    only; the training loop should subtract this from task reward.
    """

    message_dim: int = 8
    n_levels: Optional[int] = None
    latency_steps: int = 0
    loss_prob: float = 0.0
    j_per_kb: float = 0.2


class _StraightThroughQuantize(torch.autograd.Function):
    """Forward: round to discrete levels. Backward: identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, n_levels: int) -> torch.Tensor:
        # Map tanh-like [-1, 1] inputs to [0, n_levels-1]
        idx = torch.clamp(((x + 1.0) * 0.5) * (n_levels - 1), 0, n_levels - 1)
        idx = torch.round(idx)
        return idx / (n_levels - 1) * 2.0 - 1.0

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class CommChannel(nn.Module):
    """Learnable channel with bandwidth, latency, loss, and energy accounting.

    The module is stateful: it keeps a ring buffer for latency and a
    running energy counter readable via :meth:`total_energy_j`. Reset
    between episodes with :meth:`reset`.
    """

    def __init__(self, config: CommChannelConfig):
        super().__init__()
        self.config = config
        self._buffer: deque[torch.Tensor] = deque(maxlen=max(1, config.latency_steps + 1))
        self._total_energy_j = 0.0

    def reset(self) -> None:
        self._buffer.clear()
        self._total_energy_j = 0.0

    def total_energy_j(self) -> float:
        return self._total_energy_j

    def forward(self, messages: torch.Tensor) -> torch.Tensor:
        """Pass ``messages`` (``(E, message_dim)``) through the channel.

        Returns the messages that *arrive at the destinations this step*
        — after quantization, latency, and loss are applied. When latency
        is positive and the buffer is not yet full, earlier steps return
        zeros (the "no message yet" state).
        """
        cfg = self.config
        msg = messages
        if cfg.n_levels is not None:
            msg = _StraightThroughQuantize.apply(torch.tanh(msg), int(cfg.n_levels))

        # Energy accounting: bits == message_dim * log2(n_levels) approximately
        bits_per_msg = cfg.message_dim * (1.0 if cfg.n_levels is None else float(np.log2(max(2, cfg.n_levels))))
        kb = bits_per_msg * msg.shape[0] / 1024.0
        self._total_energy_j += kb * cfg.j_per_kb

        self._buffer.append(msg)
        if len(self._buffer) <= cfg.latency_steps:
            return torch.zeros_like(msg)
        delivered = self._buffer[0]

        if cfg.loss_prob > 0:
            keep = (torch.rand(delivered.shape[0], device=delivered.device) >= cfg.loss_prob).float()
            delivered = delivered * keep[:, None]

        return delivered


__all__ = ["CommChannel", "CommChannelConfig"]
