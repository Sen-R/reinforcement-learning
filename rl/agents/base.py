"""Module defines base agent class."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for agents."""

    @abstractmethod
    def action(self, state):
        """Requests desired action from the agent given state signal."""
        pass

    @abstractmethod
    def reward(self, r):
        """Sends reward signal `r` to the agent."""
        pass
