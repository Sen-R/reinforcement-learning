"""Module defines base environment class."""

from abc import ABC, abstractmethod


class Environment(ABC):
    """Base class for environments."""

    @abstractmethod
    def act(self, action):
        """Tells environment which action to perform and returns reward.

        Args:
          action: The action the agent would like to perform

        Returns:
          A numerical reward signal following the taking of this action.
        """
        pass

    @abstractmethod
    def state(self):
        """Returns environment's state signal from agent's perspective."""
        pass
