"""
Shared CTDE GNN policy base class.

CTDE = Centralized Training, Decentralized Execution. All agents share
the same policy weights (parameter sharing is the standard choice at
swarm scale; otherwise you'd need 10^6 sets of parameters). At training
time a centralized critic sees the full swarm state; at execution time
each agent only sees its own local observation.

Usage:

    actor = SharedMLPHead(in_dim=node_feat_dim + msg_dim, out_dim=action_dim)
    critic = SharedMLPHead(in_dim=global_feat_dim, out_dim=1)
    policy = GraphActorCritic(actor, critic, comm_channel=CommChannel(...))

This base class deliberately uses a generic message-passing forward pass
so new architectures (attention, transformers) can subclass without
rewriting the rollout logic.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from gossamer.learning.comm_channel import CommChannel


class SharedMLPHead(nn.Module):
    """A plain MLP used as either actor or critic head."""

    def __init__(self, in_dim: int, out_dim: int, hidden: tuple[int, ...] = (128, 128)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphActorCritic(nn.Module):
    """Parameter-shared GNN actor + centralized critic.

    The forward pass is the standard "encode nodes, pass messages,
    decode" triple. Attention or deeper message passing are future
    subclasses — this one is the minimum viable CTDE baseline.
    """

    def __init__(
        self,
        actor_head: nn.Module,
        critic_head: nn.Module,
        node_encoder: Optional[nn.Module] = None,
        edge_encoder: Optional[nn.Module] = None,
        comm_channel: Optional[CommChannel] = None,
    ):
        super().__init__()
        self.node_encoder = node_encoder or nn.Identity()
        self.edge_encoder = edge_encoder or nn.Identity()
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.comm_channel = comm_channel

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edges: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(actions, values)``.

        ``actions`` are ``(N, A)`` raw action logits/means; wrap in a
        distribution outside this class.
        ``values`` are ``(N, 1)`` per-agent state values; a centralized
        critic subclass can override to read a global feature vector.
        """
        h_nodes = self.node_encoder(node_features)
        if edges.numel() > 0:
            src = edges[:, 0]
            dst = edges[:, 1]
            # Standard MPGNN message: concatenate edge feat with sender hidden
            sender_h = h_nodes[src]
            raw_msg = torch.cat([sender_h, self.edge_encoder(edge_features)], dim=-1)
            if self.comm_channel is not None:
                # Project into the channel's message_dim via first linear layer
                # of the actor head; for stub-simplicity, truncate/pad to size.
                msg = raw_msg
                if msg.shape[-1] > self.comm_channel.config.message_dim:
                    msg = msg[..., : self.comm_channel.config.message_dim]
                elif msg.shape[-1] < self.comm_channel.config.message_dim:
                    pad = self.comm_channel.config.message_dim - msg.shape[-1]
                    msg = torch.nn.functional.pad(msg, (0, pad))
                msg = self.comm_channel(msg)
                raw_msg = msg
            # Scatter-sum into destinations
            agg = torch.zeros(num_nodes, raw_msg.shape[-1], device=h_nodes.device, dtype=h_nodes.dtype)
            agg.index_add_(0, dst, raw_msg)
        else:
            agg = torch.zeros(num_nodes, h_nodes.shape[-1], device=h_nodes.device, dtype=h_nodes.dtype)
        combined = torch.cat([h_nodes, agg], dim=-1) if agg.shape[-1] > 0 else h_nodes
        actions = self.actor_head(combined)
        values = self.critic_head(combined)
        return actions, values


__all__ = ["GraphActorCritic", "SharedMLPHead"]
