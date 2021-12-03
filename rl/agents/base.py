"""Module defines base agent class."""

from abc import ABC, abstractmethod
from typing import final


class Agent(ABC):
    """Base class for agents."""

    @final
    def action(self, state):
        """Requests desired action from the agent given state signal.

        By default it calls `_get_action_selector` to retrieve an
        `ActionSelector` instance, which is then called to select a specified
        action.
        """
        action_selector = self._get_action_selector(state)
        chosen_action = action_selector()
        self.last_state = state
        self.last_action = chosen_action
        return chosen_action

    @final
    def reward(self, reward):
        """Sends reward signal `r` to the agent."""
        self._process_reward(self.last_state, self.last_action, reward)

    @property
    @abstractmethod
    def n_actions(self):
        """Returns size of (discrete) action space known to agent."""
        pass

    @abstractmethod
    def _get_action_selector(self, state):
        """Returns (potentially stochastic) `ActionSelector` given state."""
        pass

    @abstractmethod
    def _process_reward(self, last_state, last_action, reward):
        """Function to be implemented for agent to process received reward."""
        pass
