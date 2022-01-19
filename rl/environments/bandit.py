"""Implementation of a multi-armed bandit.

This module provides an implementation of a multi-armed bandit environment.
"""

from typing import Tuple, Optional, Dict, Callable
from copy import copy
from numpy.typing import ArrayLike
import numpy as np
from .base import Environment


class MultiArmedBandit(Environment):
    """A multi-armed bandit.

    Class representing a multi-armed bandit. Pulling one of its levers
    by calling `act` yields a scalar reward. Rewards for each lever
    are normally distributed with parameters defined when the bandit
    is initialised.

    Optionally, by setting `state_updater`, it is possible to change the
    bandit's configuration following each pull of a lever (e.g. to make
    action values follow a stochastic process).

    Note: when the `reset` method is called, `state_updater` is initialised
    to a (shallow) copy of the object initially provided. Therefore it is
    assumed that a shallow copy is sufficient to freeze any internal state
    in this object.

    Args:
      means: Sequence of mean rewards for each lever
      sigma: Sequence of reward standard deviations for each lever
      random_state: `None`, `int`, `Generator` etc to initialise RNG.
      state_updater: callable that operates on means and sigmas after each
        `act` call to update bandit's internal parameters.
    """

    def __init__(
        self,
        means: ArrayLike,
        sigmas: ArrayLike,
        *,
        random_state=None,
        state_updater: Optional[
            Callable[[ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike]]
        ] = None,
    ):
        self.initial_args = {
            "means": np.array(means),
            "sigmas": np.array(sigmas),
            "random_state": random_state,
            "state_updater": state_updater,
        }
        self.reset()

    def reset(self) -> None:
        self.means = np.array(self.initial_args["means"])
        self.sigmas = np.array(self.initial_args["sigmas"])
        self._rng = np.random.default_rng(self.initial_args["random_state"])
        self.state_updater = copy(self.initial_args["state_updater"])

    @property
    def k(self) -> int:
        """Returns the number of levers for this bandit."""
        return len(self.means)

    def act(self, lever: int) -> float:
        """Returns reward for pulling lever `lever`."""
        reward = self._rng.normal(
            loc=self.means[lever], scale=self.sigmas[lever]
        )
        if self.state_updater is not None:
            self.means, self.sigmas = self.state_updater(
                self.means, self.sigmas
            )
        return reward

    def observe(self) -> None:
        """Bandit is stateless so always returns `None`."""
        return None

    def optimal_action(self) -> int:
        return int(np.argmax(self.means))

    @property
    def done(self) -> bool:
        """Non episodic environment, so always returns False."""
        return False

    @property
    def state(self) -> Dict[str, ArrayLike]:
        return {"means": self.means.copy(), "sigmas": self.sigmas.copy()}


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
