"""Implementation of a multi-armed bandit.

This module provides an implementation of a multi-armed bandit environment.
"""

import numpy as np
from .base import Environment


class MultiArmedBandit(Environment):
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


def random_bandit(k, *, mean_params, sigma_params, random_state=None):
    """Returns a randomly generated `k`-armed bandit instance.

    Args:
      k: number of levers required on the bandit to be returned
      mean_params: tuple containing location and scale parameters for
        the normal distribution from which the mean rewards (per lever) will
        be drawn
      scale_params: tuple containing the location and scale parameters for
        the normal distribution from which the reward standard deviations
        (per lever) will be drawn.
      random_state: `None`, `int`, `np.random.Generator` etc to initialise
        RNG. Note this is currently used to initialise both the RNG used
        to determine the bandit's configuration, and also to initialise
        the RNG embedded within the bandit itself.

    Returns:
        `MultiArmedBandit` instance with randomly chosen reward distributions
        for each of its `k` levers.
    """
    rng = np.random.default_rng(random_state)
    means = rng.normal(loc=mean_params[0], scale=mean_params[1], size=(k,))
    sigmas = rng.normal(loc=sigma_params[0], scale=sigma_params[1], size=(k,))
    return MultiArmedBandit(
        means=means, sigmas=sigmas, random_state=random_state
    )
