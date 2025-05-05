"""
Perception models for agents sensing their environment.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import NamedTuple, Optional


class Observation(NamedTuple):
    """
    Observation data from an agent's perspective.
    indices: array of neighbor indices perceived.
    positions: array of shape (n_neighbors, n_dims) of neighbor positions.
    velocities: optional array of neighbor velocities.
    """
    indices: np.ndarray
    positions: np.ndarray
    velocities: Optional[np.ndarray]


class BasePerception(ABC):
    """
    Abstract base class for perception models.
    """
    @abstractmethod
    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        """
        Return the Observation for a given agent_index based on full
        positions and optional velocities.
        """
        pass


class PerfectPerception(BasePerception):
    """
    Perfect perception: agent perceives all other agents (excluding self),
    with optional additive Gaussian noise.
    """
    def __init__(
        self,
        noisy: bool = False,
        pos_noise: float = 0.0,
        vel_noise: float = 0.0,
    ):
        self.noisy = noisy
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        n_agents = positions.shape[0]
        indices = np.arange(n_agents)
        mask = indices != agent_index
        indices = indices[mask]
        pos = positions[indices].copy()
        if self.noisy and self.pos_noise > 0:
            pos = pos + np.random.normal(scale=self.pos_noise, size=pos.shape)
        vel = None
        if velocities is not None:
            vel = velocities[indices].copy()
            if self.noisy and self.vel_noise > 0:
                vel = vel + np.random.normal(scale=self.vel_noise, size=vel.shape)
        return Observation(indices=indices, positions=pos, velocities=vel)


class RangePerception(BasePerception):
    """
    Range-based perception: agent perceives agents within a given radius
    (excluding self), with optional additive Gaussian noise.
    """
    def __init__(
        self,
        radius: float,
        noisy: bool = False,
        pos_noise: float = 0.0,
        vel_noise: float = 0.0,
    ):
        self.radius = radius
        self.noisy = noisy
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        agent_pos = positions[agent_index]
        # compute distances to all agents
        diffs = positions - agent_pos
        dists = np.linalg.norm(diffs, axis=1)
        all_indices = np.arange(len(positions))
        mask = (dists <= self.radius) & (all_indices != agent_index)
        indices = all_indices[mask]
        pos = positions[indices].copy()
        if self.noisy and self.pos_noise > 0:
            pos = pos + np.random.normal(scale=self.pos_noise, size=pos.shape)
        vel = None
        if velocities is not None:
            vel = velocities[indices].copy()
            if self.noisy and self.vel_noise > 0:
                vel = vel + np.random.normal(scale=self.vel_noise, size=vel.shape)
        return Observation(indices=indices, positions=pos, velocities=vel)
 
class FieldOfViewPerception(BasePerception):
    """
    Cone-based perception: agent perceives agents within a given radius
    and within a field-of-view (FOV) angle centered on its heading.
    Requires velocities to determine heading direction.
    """
    def __init__(
        self,
        radius: float,
        fov: float,
        noisy: bool = False,
        pos_noise: float = 0.0,
        vel_noise: float = 0.0,
    ):
        self.radius = radius
        self.fov = fov
        self.noisy = noisy
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        if velocities is None:
            raise ValueError("FieldOfViewPerception requires velocities to determine heading")
        # Agent state
        agent_pos = positions[agent_index]
        agent_vel = velocities[agent_index]
        # Heading unit vector
        speed = np.linalg.norm(agent_vel)
        if speed > 0:
            heading = agent_vel / speed
        else:
            heading = None
        # Compute relative positions and distances
        diffs = positions - agent_pos
        dists = np.linalg.norm(diffs, axis=1)
        all_idx = np.arange(len(positions))
        # Radius mask, exclude self
        mask = (dists <= self.radius) & (all_idx != agent_index)
        idxs = all_idx[mask]
        rel = diffs[mask]
        # FOV mask
        if heading is not None:
            # cos(theta) threshold
            cos_th = np.cos(self.fov / 2)
            # normalize rel vectors
            rel_norm = rel / dists[mask][:, None]
            cos_vals = rel_norm.dot(heading)
            fov_mask = cos_vals >= cos_th
            idxs = idxs[fov_mask]
            rel = rel[fov_mask]
            dists_masked = dists[mask][fov_mask]
        else:
            dists_masked = dists[mask]
        # Positions of neighbors
        pos = positions[idxs].copy()
        if self.noisy and self.pos_noise > 0:
            pos = pos + np.random.normal(scale=self.pos_noise, size=pos.shape)
        vel = None
        if velocities is not None:
            vel = velocities[idxs].copy()
            if self.noisy and self.vel_noise > 0:
                vel = vel + np.random.normal(scale=self.vel_noise, size=vel.shape)
        return Observation(indices=idxs, positions=pos, velocities=vel)

from collections import deque

class DelayPerception(BasePerception):
    """
    Perception wrapper that delays observations by a fixed number of steps.
    """
    def __init__(self, base: BasePerception, delay_steps: int):
        self.base = base
        self.delay_steps = delay_steps
        self.history = deque(maxlen=delay_steps + 1)

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        obs = self.base.perceive(agent_index, positions, velocities)
        self.history.append(obs)
        # Return the observation from delay_steps ago (or earliest available)
        return self.history[0]

class PacketLossPerception(BasePerception):
    """
    Perception wrapper that randomly drops perceived neighbors with a given probability.
    """
    def __init__(self, base: BasePerception, loss_prob: float):
        self.base = base
        self.loss_prob = loss_prob

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        obs = self.base.perceive(agent_index, positions, velocities)
        idxs = obs.indices
        if idxs.size == 0:
            return obs
        # Sample drop mask
        keep_mask = np.random.rand(len(idxs)) >= self.loss_prob
        kept = keep_mask.nonzero()[0]
        new_idxs = idxs[keep_mask]
        new_pos = obs.positions[keep_mask]
        new_vel = None
        if obs.velocities is not None:
            new_vel = obs.velocities[keep_mask]
        return Observation(indices=new_idxs, positions=new_pos, velocities=new_vel)
        
class BearingOnlyPerception(BasePerception):
    """
    Perception wrapper that only provides bearing (angle) to each neighbor
    relative to the agent's heading. Positions and velocities are not returned.
    Requires velocities to determine heading; if speed is zero, uses +x axis.
    """
    def __init__(self, base: BasePerception):
        self.base = base

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        obs = self.base.perceive(agent_index, positions, velocities)
        # Determine agent heading
        if velocities is None:
            raise ValueError("BearingOnlyPerception requires velocities for heading")
        vel = velocities[agent_index]
        speed = np.linalg.norm(vel)
        if speed > 0:
            heading_angle = np.arctan2(vel[1], vel[0])
        else:
            heading_angle = 0.0
        # Compute bearings for each neighbor
        rel = obs.positions - positions[agent_index]
        angles = np.arctan2(rel[:, 1], rel[:, 0]) - heading_angle
        # Normalize to [-pi, pi]
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # Return angles as positions, drop velocities
        return Observation(indices=obs.indices, positions=angles, velocities=None)

import random

class IntermittentBlindSpotPerception(BasePerception):
    """
    Perception wrapper that intermittently creates a blind spot: with a given probability,
    a random angular sector is occluded so neighbors within that sector are not perceived.
    Requires velocities to determine heading; if speed=0, uses +x axis.
    """
    def __init__(
        self,
        base: BasePerception,
        blind_width: float,
        prob_active: float,
    ):
        self.base = base
        self.blind_width = blind_width
        self.prob_active = prob_active

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        obs = self.base.perceive(agent_index, positions, velocities)
        if obs.indices.size == 0:
            return obs
        if random.random() >= self.prob_active:
            return obs
        # Determine agent heading
        if velocities is None:
            raise ValueError("IntermittentBlindSpotPerception requires velocities for heading")
        vel = velocities[agent_index]
        speed = np.linalg.norm(vel)
        if speed > 0:
            heading_angle = np.arctan2(vel[1], vel[0])
        else:
            heading_angle = 0.0
        # Sample blind spot center in [-pi, pi]
        center = random.uniform(-np.pi, np.pi)
        half = self.blind_width / 2.0
        # Compute relative angles
        rel = obs.positions - positions[agent_index]
        angles = np.arctan2(rel[:, 1], rel[:, 0]) - heading_angle
        angles = (angles + np.pi) % (2 * np.pi) - np.pi
        # Mask out those within blind sector
        mask = np.logical_or(
            (angles < center - half),
            (angles > center + half)
        )
        new_idx = obs.indices[mask]
        new_pos = obs.positions[mask]
        new_vel = None
        if obs.velocities is not None:
            new_vel = obs.velocities[mask]
        return Observation(indices=new_idx, positions=new_pos, velocities=new_vel)

class StochasticDelayPerception(BasePerception):
    """
    Perception wrapper that returns observations delayed by a random number of steps.
    Delay sampled via delay_sampler() or uniform integer in [0, max_delay_steps].
    """
    def __init__(
        self,
        base: BasePerception,
        max_delay_steps: int,
        delay_sampler=None,
    ):
        self.base = base
        self.max_delay_steps = max_delay_steps
        self.delay_sampler = delay_sampler
        self.history = deque(maxlen=max_delay_steps + 1)

    def perceive(
        self,
        agent_index: int,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
    ) -> Observation:
        obs = self.base.perceive(agent_index, positions, velocities)
        self.history.append(obs)
        # Determine delay
        if self.delay_sampler is not None:
            d = self.delay_sampler()
        else:
            d = random.randint(0, self.max_delay_steps)
        # Return observation from d steps ago (or earliest)
        if d < len(self.history):
            return self.history[-(d + 1)]
        return self.history[0]