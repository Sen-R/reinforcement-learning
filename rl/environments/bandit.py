"""Implementation of a multi-armed bandit.

This module provides an implementation of a multi-armed bandit environment.
"""

import numpy as np
from .base import Environment


class MultiArmBandit(Environment):
    """A multi-armed bandit.

    Class representing a multi-armed bandit. Pulling one of its levers
    by calling `act` yields a scalar reward. Rewards for each lever
    are normally distributed with parameters defined when the bandit
    is initialised.

    Args:
      means: Sequence of mean rewards for each lever
      sigma: Sequence of reward standard deviations for each lever
      random_state: `None`, `int`, `Generator` etc to initialise RNG.
    """

    def __init__(self, means, sigmas, random_state=None):
        self.means = means
        self.sigmas = sigmas
        self._rng = np.random.default_rng(random_state)

    @property
    def k(self):
        """Returns the number of levers for this bandit."""
        return len(self.means)

    def act(self, lever):
        """Returns reward for pulling lever `lever`."""
        return self._rng.normal(
            loc=self.means[lever], scale=self.sigmas[lever]
        )

    def state(self):
        """Bandit is stateless so always returns `None`."""
        return None
