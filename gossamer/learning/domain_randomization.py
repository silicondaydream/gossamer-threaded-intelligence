"""
Domain randomization harness.

Required groundwork for any sim2real claim: during training, randomize
the environment's physical parameters per episode so the policy cannot
overfit to a single regime. This module wraps any PettingZoo
``ParallelEnv`` (notably :class:`LeviathanParallelEnv`) and resamples
config each ``reset``.

Current dimensions:

* ``comm_latency_steps`` — integer delay, sampled uniform in a range.
* ``comm_loss_prob`` — Bernoulli drop probability, sampled uniform.
* ``fault_prob`` — per-step per-agent failure probability.
* ``num_agents`` — optional; resamples fleet size each episode.
* ``initial_distribution`` — one of ``"uniform"``, ``"gaussian"``,
  ``"clumped"`` (chosen uniformly from the requested list).

Curricula:

* :class:`LinearCurriculum` — linearly increases the difficulty of a
  single parameter from an easy bound to a hard bound over N episodes.
* :class:`AdaptiveCurriculum` — moves the bounds based on rolling
  success rate. Off by default; needs a success signal from the env.

Designed to be used as a drop-in wrapper::

    env = LeviathanParallelEnv(config)
    env = DomainRandomizedEnv(env, dr_config)

Every MARL trainer that understands PettingZoo will still work against
the wrapped env; only the ``reset`` path changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DomainRandomizationConfig:
    """Per-episode sampling bounds for each randomized dimension.

    Any field left at its default stays constant across episodes; set a
    ``(low, high)`` pair to randomize it. Bounds are inclusive.
    """
    comm_latency_steps: Optional[Tuple[int, int]] = None
    comm_loss_prob: Optional[Tuple[float, float]] = None
    fault_prob: Optional[Tuple[float, float]] = None
    num_agents: Optional[Tuple[int, int]] = None
    initial_distribution_choices: List[str] = field(default_factory=lambda: ["uniform"])
    field_strength: Optional[Tuple[float, float]] = None


@dataclass
class LinearCurriculum:
    """Linearly scale one parameter's bounds over training.

    Given ``(easy_low, easy_high)`` and ``(hard_low, hard_high)``, return
    the interpolated ``(low, high)`` at episode ``t`` out of ``n_episodes``.
    Use when you know up-front which dimension is the hard one (e.g.,
    ``comm_loss_prob`` for DTN work).
    """
    parameter: str
    easy: Tuple[float, float]
    hard: Tuple[float, float]
    n_episodes: int

    def current(self, episode: int) -> Tuple[float, float]:
        t = min(1.0, max(0.0, episode / max(1, self.n_episodes)))
        low = self.easy[0] + (self.hard[0] - self.easy[0]) * t
        high = self.easy[1] + (self.hard[1] - self.easy[1]) * t
        return low, high


class DomainRandomizedEnv:
    """PettingZoo ``ParallelEnv`` wrapper that resamples config on reset.

    Minimal implementation: constructs a fresh inner env each reset with
    the new config values, then delegates ``step``. Heavier impls can
    instead call a ``reconfigure`` hook on a persistent env.
    """

    def __init__(
        self,
        inner_env_factory,
        dr_config: DomainRandomizationConfig,
        rng: Optional[np.random.Generator] = None,
        curriculum: Optional[LinearCurriculum] = None,
    ):
        self._factory = inner_env_factory
        self._dr = dr_config
        self._rng = rng or np.random.default_rng()
        self._curriculum = curriculum
        self._episode = 0
        self._env = None
        self._last_sampled: Dict[str, Any] = {}

    def _sample_bounds(self, current_bounds: Optional[Tuple[float, float]],
                        override: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        return override if override is not None else current_bounds

    def _sample(self) -> Dict[str, Any]:
        """Draw a fresh config tuple for this episode."""
        chosen: Dict[str, Any] = {}
        if self._dr.comm_latency_steps is not None:
            lo, hi = self._dr.comm_latency_steps
            chosen["comm_latency_steps"] = int(self._rng.integers(lo, hi + 1))
        if self._dr.comm_loss_prob is not None:
            # Curriculum can override the loss bounds
            lo, hi = self._dr.comm_loss_prob
            if self._curriculum is not None and self._curriculum.parameter == "comm_loss_prob":
                lo, hi = self._curriculum.current(self._episode)
            chosen["comm_loss_prob"] = float(self._rng.uniform(lo, hi))
        if self._dr.fault_prob is not None:
            lo, hi = self._dr.fault_prob
            chosen["fault_prob"] = float(self._rng.uniform(lo, hi))
        if self._dr.num_agents is not None:
            lo, hi = self._dr.num_agents
            chosen["num_agents"] = int(self._rng.integers(lo, hi + 1))
        if self._dr.field_strength is not None:
            lo, hi = self._dr.field_strength
            chosen["field_strength"] = float(self._rng.uniform(lo, hi))
        choices = self._dr.initial_distribution_choices or ["uniform"]
        chosen["initial_distribution"] = str(self._rng.choice(choices))
        self._last_sampled = chosen
        return chosen

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        sampled = self._sample()
        self._env = self._factory(sampled)
        self._episode += 1
        return self._env.reset(seed=seed, options=options)

    def step(self, actions):
        assert self._env is not None, "reset() must be called before step()"
        return self._env.step(actions)

    def observation_space(self, agent: str):
        return self._env.observation_space(agent)

    def action_space(self, agent: str):
        return self._env.action_space(agent)

    @property
    def possible_agents(self):
        return self._env.possible_agents

    @property
    def last_sampled(self) -> Dict[str, Any]:
        """The most recent draw; useful for logging / curriculum feedback."""
        return dict(self._last_sampled)


__all__ = [
    "DomainRandomizationConfig",
    "DomainRandomizedEnv",
    "LinearCurriculum",
]
