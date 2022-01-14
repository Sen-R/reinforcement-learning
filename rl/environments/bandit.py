"""Implementation of a multi-armed bandit.

This module provides an implementation of a multi-armed bandit environment.
"""

from typing import Tuple, Optional
from numpy.typing import ArrayLike
import numpy as np
from .base import Environment


class MultiArmedBandit(Environment):
    """A multi-armed bandit.

    Class representing a multi-armed bandit. Pulling one of its levers
    by calling `act` yields a scalar reward. Rewards for each lever
    are normally distributed with parameters defined when the bandit
    is initialised.

    Optionally, by setting `random_walk_params`, the bandit's expected
    rewards (as stored in the `means` attribute) can take a random walk,
    changing after each `act` step that is taken.

    Args:
      means: Sequence of mean rewards for each lever
      sigma: Sequence of reward standard deviations for each lever
      random_state: `None`, `int`, `Generator` etc to initialise RNG.
      random_walk_params: `(mean, sigma)` tuple parameterising independent
        normal variates to additively apply to each lever mean after every
        `act` step.
    """

    def __init__(
        self,
        means: ArrayLike,
        sigmas: ArrayLike,
        *,
        random_state=None,
        random_walk_params: Optional[Tuple[float, float]] = None,
    ):
        self.means = np.array(means)
        self.sigmas = np.array(sigmas)
        self.random_walk_params = random_walk_params
        self.reset(random_state)

    @property
    def k(self) -> int:
        """Returns the number of levers for this bandit."""
        return len(self.means)

    def act(self, lever: int) -> float:
        """Returns reward for pulling lever `lever`."""
        reward = self._rng.normal(
            loc=self.means[lever], scale=self.sigmas[lever]
        )
        if self.random_walk_params is not None:
            self.means += self._rng.normal(
                loc=self.random_walk_params[0],
                scale=self.random_walk_params[1],
                size=self.means.shape,
            )
        return reward

    def observe(self) -> None:
        """Bandit is stateless so always returns `None`."""
        return None

    def optimal_action(self) -> int:
        return int(np.argmax(self.means))

    def reset(self, random_state=None) -> None:
        self._rng = np.random.default_rng(random_state)

    @property
    def done(self) -> bool:
        """Non episodic environment, so always returns False."""
        return False


def random_bandit(
    k,
    *,
    mean_params: Tuple[float, float],
    sigma_params: Tuple[float, float],
    random_state=None,
) -> MultiArmedBandit:
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
        RNG. Note the RNG is first used to initialise the bandit's
        configuration and then passed to the bandit as the RNG for determining
        rewards.

    Returns:
        `MultiArmedBandit` instance with randomly chosen reward distributions
        for each of its `k` levers.
    """
    rng = np.random.default_rng(random_state)
    means = rng.normal(loc=mean_params[0], scale=mean_params[1], size=(k,))
    sigmas = rng.normal(loc=sigma_params[0], scale=sigma_params[1], size=(k,))
    return MultiArmedBandit(means=means, sigmas=sigmas, random_state=rng)
