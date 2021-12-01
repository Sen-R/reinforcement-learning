"""Implementation of agents that take random actions."""

import numpy as np
from .base import Agent


class DiscreteRandomAgent(Agent):
    """Randomly acting agent for discrete action spaces."""

    def __init__(self, n_actions, random_state=None):
        self._n_actions = n_actions
        self._rng = np.random.default_rng(random_state)

    @property
    def n_actions(self):
        return self._n_actions

    def action(self, state=None):
        """Note state signal can be `None` if desired. (E.g. for multi-armed
        bandit problem.)
        """
        return self._rng.integers(self.n_actions)

    def reward(self, r):
        """No-op in the case of a random agent."""
        pass
