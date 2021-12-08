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
    def observe(self):
        """Returns environment's state signal from agent's perspective."""
        pass

    @abstractmethod
    def reset(self, random_state=None) -> None:
        """Resets current environment (e.g. to avoid making from scratch)."""
        pass

    @property
    @abstractmethod
    def done(self) -> bool:
        """Whether terminal state has been reached."""
        pass
