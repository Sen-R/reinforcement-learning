"""Implementation of agents that take random actions."""

from .base import Agent
from .action_selector import UniformDiscreteActionSelector


class DiscreteRandomAgent(Agent):
    """Randomly acting agent for discrete action spaces."""

    def __init__(self, n_actions, random_state=None):
        self._n_actions = n_actions
        self._action_selector = UniformDiscreteActionSelector(
            self._n_actions, random_state=random_state
        )

    @property
    def n_actions(self):
        return self._n_actions

    def _get_action_selector(self, _):
        return self._action_selector

    def _process_reward(self, last_state, last_action, reward):
        """No-op in the case of a random agent."""
        pass
