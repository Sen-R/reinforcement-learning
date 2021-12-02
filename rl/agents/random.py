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

    def action(self, state=None):
        """Allows state not to be specified, as it doesn't matter for
        a random agent."""
        return super().action(state)

    def reward(self, r):
        """No-op in the case of a random agent."""
        pass

    def _get_action_selector(self, _):
        return self._action_selector
