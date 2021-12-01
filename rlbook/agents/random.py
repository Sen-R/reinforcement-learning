"""Implementation of agents that take random actions."""

import numpy as np


class DiscreteRandomAgent:
    """Randomly acting agent for discrete action spaces."""

    def __init__(self, n_actions, random_state=None):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(random_state)

    def action(self, state=None):
        """Requests desired action from the agent given state signal.

        Note state signal can be `None` if desired. (E.g. for multi-armed
        bandit problem.)
        """
        return self._rng.integers(self.n_actions)

    def reward(self, r):
        """Sends a reward signal to the agent.

        No-op in the case of a random agent."""
        pass
