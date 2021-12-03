"""Module defines base agent class."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Base class for agents."""

    def action(self, state):
        """Requests desired action from the agent given state signal.

        By default it calls `_get_action_selector` to retrieve an
        `ActionSelector` instance, which is then called to select a specified
        action.
        """
        action_selector = self._get_action_selector(state)
        chosen_action = action_selector()
        self.last_action = chosen_action
        return chosen_action

    @abstractmethod
    def reward(self, r):
        """Sends reward signal `r` to the agent."""
        pass

    @property
    @abstractmethod
    def n_actions(self):
        """Returns size of (discrete) action space known to agent."""
        pass

    @abstractmethod
    def _get_action_selector(self, state):
        """Returns (potentially stochastic) `ActionSelector` given state."""
        pass
