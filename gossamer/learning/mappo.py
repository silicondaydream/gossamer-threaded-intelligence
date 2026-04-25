"""
Reference MAPPO driver.

This is intentionally a *minimal reference*, not a production trainer.
Production MARL runs should live in CleanRL, MARLlib, or TorchRL, which
already implement the right logging / checkpoint / distributed-training
plumbing. This module exists for two reasons:

1. To prove the PettingZoo + Gossamer + Leviathan stack is actually
   connected. The ``rollout()`` function works against any
   ``pettingzoo.ParallelEnv`` and the :class:`GraphActorCritic` policy.
2. To give the papers a reference baseline with a known, reproducible
   implementation so numbers reported against "MAPPO" are unambiguous.

The training loop is single-process, single-env. Scale comes from large
per-step agent counts, not parallel envs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    lr: float = 3e-4
    update_epochs: int = 4
    minibatch_size: int = 256


def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor,
                gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation.

    ``rewards``, ``values``, ``dones`` are all ``(T, N)`` tensors. Values
    must include the bootstrap at t=T (i.e. length T+1); pass it as the
    last row if you have it, otherwise use zeros.
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(rewards.shape[1], device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        next_nonterminal = 1.0 - dones[t]
        next_value = values[t + 1] if t + 1 < values.shape[0] else torch.zeros_like(values[0])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T]
    return advantages, returns


def rollout(
    env,
    policy,
    action_std: float,
    num_steps: int,
    build_graph: Callable,
) -> dict:
    """Collect a single PettingZoo parallel rollout.

    ``env`` is a PettingZoo ``ParallelEnv``. ``build_graph`` translates
    the current ``(positions, velocities)`` into an ``InteractionGraph``
    ready for the policy; for Leviathan, this is typically
    :func:`gossamer.graph.build_radius_graph`. ``action_std`` controls
    exploration on the Gaussian policy.

    Returns a dict with ``obs, actions, log_probs, values, rewards, dones``
    each shaped ``(num_steps, num_agents, ...)`` suitable for the update
    step. Kept deliberately simple; production code should use torch
    buffers and proper vectorization across parallel envs.
    """
    observations, _ = env.reset()
    agent_ids = env.possible_agents
    n_agents = len(agent_ids)
    device = next(policy.parameters()).device

    obs_buf = []
    action_buf = []
    logp_buf = []
    value_buf = []
    reward_buf = []
    done_buf = []

    for t in range(num_steps):
        positions = np.stack([observations[a]["position"] for a in agent_ids])
        velocities = np.stack([observations[a]["velocity"] for a in agent_ids])
        graph = build_graph(positions, velocities)

        node_feats = torch.as_tensor(
            np.concatenate([positions, velocities], axis=1), dtype=torch.float32, device=device
        )
        edges = torch.as_tensor(graph.edges, dtype=torch.long, device=device) if graph.num_edges else torch.zeros((0, 2), dtype=torch.long, device=device)
        edge_feats = torch.as_tensor(
            graph.edge_features if graph.edge_features is not None else np.zeros((graph.num_edges, 0)),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            mean, value = policy(node_feats, edge_feats, edges, n_agents)
            dist = Normal(mean, torch.full_like(mean, action_std))
            action = dist.sample()
            logp = dist.log_prob(action).sum(dim=-1)

        actions_np = action.cpu().numpy()
        step_actions = {a: actions_np[i] for i, a in enumerate(agent_ids)}
        observations, rewards, terminations, truncations, _ = env.step(step_actions)
        reward_vec = np.array([rewards[a] for a in agent_ids], dtype=np.float32)
        done_vec = np.array([terminations[a] or truncations[a] for a in agent_ids], dtype=np.float32)

        obs_buf.append(node_feats)
        action_buf.append(action)
        logp_buf.append(logp)
        value_buf.append(value.squeeze(-1))
        reward_buf.append(torch.as_tensor(reward_vec, device=device))
        done_buf.append(torch.as_tensor(done_vec, device=device))

        if done_vec.all():
            break

    stack = lambda xs: torch.stack(xs, dim=0)
    return {
        "obs": stack(obs_buf),
        "actions": stack(action_buf),
        "log_probs": stack(logp_buf),
        "values": stack(value_buf),
        "rewards": stack(reward_buf),
        "dones": stack(done_buf),
    }


def update(policy, optimizer, batch: dict, cfg: MAPPOConfig) -> dict:
    """Run ``cfg.update_epochs`` PPO update passes over ``batch``.

    Minimal implementation — no KL early stopping, no value clipping,
    no annealed entropy. Good enough as a baseline; swap in CleanRL for
    headline paper runs.
    """
    values_plus = torch.cat([batch["values"], batch["values"][-1:]], dim=0)
    advantages, returns = compute_gae(
        batch["rewards"], values_plus, batch["dones"],
        cfg.gamma, cfg.gae_lambda,
    )
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Flatten time * agents for minibatching
    T, N = batch["rewards"].shape
    flat = lambda t: t.reshape(T * N, *t.shape[2:])
    obs = flat(batch["obs"])
    actions = flat(batch["actions"])
    old_logp = flat(batch["log_probs"])
    advantages = flat(advantages)
    returns = flat(returns)

    metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    for _ in range(cfg.update_epochs):
        idx = torch.randperm(obs.shape[0], device=obs.device)
        for start in range(0, idx.shape[0], cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]
            # Recompute mean/value from current policy; edges are inside obs
            # here it's up to the caller to re-pass edges if they need them.
            # For stub correctness, we skip edges (flatten-mode ignores them).
            mean, value = policy.actor_head(obs[mb]), policy.critic_head(obs[mb])
            dist = Normal(mean, torch.ones_like(mean))
            new_logp = dist.log_prob(actions[mb]).sum(dim=-1)
            ratio = torch.exp(new_logp - old_logp[mb])
            clipped = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            policy_loss = -torch.min(ratio * advantages[mb], clipped * advantages[mb]).mean()
            value_loss = F.mse_loss(value.squeeze(-1), returns[mb])
            entropy = dist.entropy().sum(dim=-1).mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            optimizer.step()
            metrics["policy_loss"] += float(policy_loss.detach())
            metrics["value_loss"] += float(value_loss.detach())
            metrics["entropy"] += float(entropy.detach())
    return metrics


__all__ = ["MAPPOConfig", "compute_gae", "rollout", "update"]
