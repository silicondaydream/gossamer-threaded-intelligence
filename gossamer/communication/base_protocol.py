"""
Base classes for communication protocols between agents.
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseProtocol(ABC):
    """
    Abstract base class for communication protocols.

    Parameters:
        adjacency: 2D array-like adjacency matrix (n_agents x n_agents)
    """
    def __init__(self, adjacency):
        A = np.asarray(adjacency, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        self.adjacency = A
        self.n = A.shape[0]
        # Precompute neighbor lists
        self.neighbors = {
            i: list(np.nonzero(A[i])[0]) for i in range(self.n)
        }

    @abstractmethod
    def step(self, messages):
        """
        Perform one communication step.

        Parameters:
            messages: dict mapping agent index to iterable of messages

        Returns:
            new_messages: dict mapping agent index to set of received messages
        """
        pass