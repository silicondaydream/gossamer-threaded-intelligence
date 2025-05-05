"""
Adapter for integrating with the Leviathan simulator.
Defines how Gossamer agents interact with a Leviathan environment.
"""
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple


class LeviathanInterface:
    """
    Wraps a Leviathan environment to provide a Gossamer-like simulator API.

    The Leviathan environment is expected to implement:
      - reset() -> Dict with 'positions' (ndarray) and optional 'velocities' (ndarray)
      - step(actions: Optional[Dict[int, Any]]) -> Dict with 'positions', optional 'velocities'
      - compute_metrics() -> Dict[str, float] of simulation metrics

    Example:
        env = LeviathanEnv(...)
        adapter = LeviathanInterface(env)
        pos, vel = adapter.step()
        metrics = adapter.metrics()
        adapter.run(100, callback=...)
    """
    def __init__(self, env: Any):
        self.env = env
        # Initialize environment state
        init_state = self.env.reset()
        self.positions: np.ndarray = init_state['positions']
        self.velocities: Optional[np.ndarray] = init_state.get('velocities')

    def step(
        self,
        actions: Optional[Dict[int, Any]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Advance the simulation by one step.

        Parameters:
            actions: Optional mapping from agent index to action data.

        Returns:
            positions: ndarray of shape (n_agents, n_dims)
            velocities: optional ndarray of same shape
        """
        out = self.env.step(actions)
        self.positions = out['positions']
        self.velocities = out.get('velocities')
        return self.positions, self.velocities

    def metrics(self) -> Dict[str, float]:
        """
        Retrieve current simulation metrics from the Leviathan environment.
        """
        return self.env.compute_metrics()

    def run(
        self,
        steps: int,
        callback: Optional[Callable[[int, np.ndarray, Optional[np.ndarray], Dict[str, float]], None]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Run the simulation for a given number of steps.

        Parameters:
            steps: number of simulation steps
            callback: optional function(step, positions, velocities, metrics)

        Returns:
            Final positions and velocities after running.
        """
        last_pos, last_vel = self.positions, self.velocities
        for i in range(steps):
            last_pos, last_vel = self.step()
            metrics = self.metrics()
            if callback:
                callback(i, last_pos, last_vel, metrics)
        return last_pos, last_vel