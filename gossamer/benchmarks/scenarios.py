"""
Canonical benchmark scenarios.

Each scenario is a small class with three responsibilities:

* :meth:`init_state` — generate the initial positions and velocities
  deterministically from a seed.
* :meth:`step_reward` — per-step reward vector (N,) given the current
  (pos, vel) and the previous state.
* :meth:`terminal_metric` — a scalar "success number" reported in the
  leaderboard (e.g., "final coverage ratio", "steps to rendezvous").

Scenarios intentionally do not manage the physics — the harness drives
either the Leviathan engine (for realism) or a local in-NumPy stepper
(for unit tests). All state is ``np.ndarray``; everything scales.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ScenarioContext:
    """Runtime parameters provided to the scenario at each step."""
    step: int
    total_steps: int
    dt: float


class Scenario(ABC):
    """Base class for a benchmark scenario."""

    name: str = "base"
    recommended_agents: Tuple[int, int] = (100, 10_000)
    success_criterion: str = ""  # plain-English, used in the leaderboard

    @abstractmethod
    def init_state(self, rng: np.random.Generator, num_agents: int, bound: float
                   ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    @abstractmethod
    def step_reward(self, pos: np.ndarray, vel: np.ndarray,
                    prev_pos: np.ndarray, prev_vel: np.ndarray,
                    ctx: ScenarioContext) -> np.ndarray:
        ...

    @abstractmethod
    def terminal_metric(self, trajectory: list) -> float:
        ...


# ---- Concrete scenarios ----


class DispersalScenario(Scenario):
    name = "dispersal"
    success_criterion = "final mean nearest-neighbor distance / bound"

    def init_state(self, rng, num_agents, bound):
        # Start clumped near origin
        pos = rng.normal(scale=bound * 0.01, size=(num_agents, 3))
        vel = np.zeros_like(pos)
        return pos, vel

    def step_reward(self, pos, vel, prev_pos, prev_vel, ctx):
        # Per-agent reward = distance to 3rd nearest neighbor (proxy for spread)
        if pos.shape[0] < 4:
            return np.zeros(pos.shape[0])
        # Sample subset for O(N*k) — full pairwise is too slow at scale
        n = pos.shape[0]
        k = min(3, n - 1)
        # Simple but correct: use argpartition
        diffs = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(d, np.inf)
        part = np.partition(d, k, axis=1)
        return part[:, k - 1]

    def terminal_metric(self, trajectory):
        if not trajectory:
            return 0.0
        pos = trajectory[-1]["pos"]
        if pos.shape[0] < 2:
            return 0.0
        diffs = pos[:, None, :] - pos[None, :, :]
        d = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(d, np.inf)
        return float(d.min(axis=1).mean())


class RendezvousScenario(Scenario):
    name = "rendezvous"
    success_criterion = "final cohesion (lower is better)"

    def init_state(self, rng, num_agents, bound):
        pos = rng.uniform(-bound, bound, size=(num_agents, 3))
        vel = np.zeros_like(pos)
        return pos, vel

    def step_reward(self, pos, vel, prev_pos, prev_vel, ctx):
        # Negative distance-to-centroid: pull together
        centroid = pos.mean(axis=0, keepdims=True)
        return -np.linalg.norm(pos - centroid, axis=1)

    def terminal_metric(self, trajectory):
        if not trajectory:
            return float("inf")
        pos = trajectory[-1]["pos"]
        centroid = pos.mean(axis=0, keepdims=True)
        return float(np.linalg.norm(pos - centroid, axis=1).mean())


class CoverageScenario(Scenario):
    name = "coverage"
    success_criterion = "unique cells visited / total cells"
    recommended_agents: Tuple[int, int] = (500, 50_000)

    def __init__(self, grid_resolution: float = 5.0):
        self.grid_resolution = grid_resolution
        self._visited: set = set()
        self._bound: float = 100.0

    def init_state(self, rng, num_agents, bound):
        self._bound = bound
        self._visited = set()
        self._cov_w = max(1, int(2 * bound / self.grid_resolution))
        pos = rng.uniform(-bound, bound, size=(num_agents, 3))
        vel = rng.normal(scale=0.1, size=(num_agents, 3))
        return pos, vel

    def step_reward(self, pos, vel, prev_pos, prev_vel, ctx):
        # Per-agent reward = 1 if this agent visited a fresh cell this step
        b = self._bound
        cells_x = np.clip(((pos[:, 0] + b) / (2 * b)) * self._cov_w, 0, self._cov_w - 1).astype(int)
        cells_y = np.clip(((pos[:, 1] + b) / (2 * b)) * self._cov_w, 0, self._cov_w - 1).astype(int)
        r = np.zeros(pos.shape[0])
        for i in range(pos.shape[0]):
            key = (int(cells_y[i]), int(cells_x[i]))
            if key not in self._visited:
                self._visited.add(key)
                r[i] = 1.0
        return r

    def terminal_metric(self, trajectory):
        total_cells = self._cov_w * self._cov_w
        return float(len(self._visited)) / float(max(1, total_cells))


class LeaderFollowerScenario(Scenario):
    name = "leader_follower"
    success_criterion = "mean follower distance to leader path"

    def __init__(self, leader_amplitude: float = 50.0):
        self.leader_amplitude = leader_amplitude

    def init_state(self, rng, num_agents, bound):
        pos = rng.normal(scale=bound * 0.1, size=(num_agents, 3))
        pos[0] = np.zeros(3)  # leader at origin
        vel = np.zeros_like(pos)
        return pos, vel

    def step_reward(self, pos, vel, prev_pos, prev_vel, ctx):
        # Followers rewarded for proximity to leader (agent 0)
        leader = pos[0]
        r = -np.linalg.norm(pos - leader, axis=1)
        r[0] = 0.0  # leader gets zero (exogenous)
        return r

    def terminal_metric(self, trajectory):
        if not trajectory:
            return float("inf")
        dists = []
        for snap in trajectory:
            p = snap["pos"]
            dists.append(float(np.linalg.norm(p[1:] - p[0], axis=1).mean()))
        return float(np.mean(dists))


class ByzantineScenario(Scenario):
    """Adversary injection: some fraction of agents emit garbage intents.

    The scenario itself doesn't corrupt state — it just wraps a base
    scenario (here: rendezvous) and reports how much the metric degrades
    vs a clean run. The corruption is modeled by the harness as
    randomized actions for the marked agents.
    """
    name = "byzantine"
    success_criterion = "rendezvous metric under k% byzantine agents"

    def __init__(self, byzantine_fraction: float = 0.1):
        self.byzantine_fraction = byzantine_fraction
        self._inner = RendezvousScenario()

    def init_state(self, rng, num_agents, bound):
        pos, vel = self._inner.init_state(rng, num_agents, bound)
        k = int(num_agents * self.byzantine_fraction)
        self.byzantine_indices = rng.choice(num_agents, size=k, replace=False) if k > 0 else np.array([], dtype=int)
        return pos, vel

    def step_reward(self, pos, vel, prev_pos, prev_vel, ctx):
        return self._inner.step_reward(pos, vel, prev_pos, prev_vel, ctx)

    def terminal_metric(self, trajectory):
        return self._inner.terminal_metric(trajectory)


ALL_SCENARIOS = {
    "dispersal": DispersalScenario,
    "rendezvous": RendezvousScenario,
    "coverage": CoverageScenario,
    "leader_follower": LeaderFollowerScenario,
    "byzantine": ByzantineScenario,
}


__all__ = [
    "ALL_SCENARIOS",
    "ByzantineScenario",
    "CoverageScenario",
    "DispersalScenario",
    "LeaderFollowerScenario",
    "RendezvousScenario",
    "Scenario",
    "ScenarioContext",
]
