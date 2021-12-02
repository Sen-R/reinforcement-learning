from abc import ABC, abstractmethod
import numpy as np


class ActionSelector(ABC):
    @abstractmethod
    def __call__(self):
        """Returns selected action."""
        pass


class DeterministicActionSelector(ActionSelector):
    """Deterministic action selector.

    Always returns the specified action when called.

    Args:
      chosen_action: action to return when called.
    """

    def __init__(self, chosen_action):
        self.chosen_action = chosen_action

    def __call__(self):
        return self.chosen_action


class UniformDiscreteActionSelector(ActionSelector):
    """Uniform discrete action selector.

    Picks an action from a discrete action space (zero-indexed) of
    size `n_actions` uniformly at random.

    Args:
      n_actions: number of actions to choose from
      random_state: `None`, `int`, `np.random.Generator`, etc for initialising
        the random number generator.
    """

    def __init__(self, n_actions, *, random_state=None):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(random_state)

    def __call__(self):
        return self._rng.integers(self.n_actions)


class NoisyActionSelector(ActionSelector):
    def __init__(
        self, epsilon, preferred_selector, noise_selector, *, random_state=None
    ):
        self.epsilon = epsilon
        self.preferred = preferred_selector
        self.noise = noise_selector
        self._rng = np.random.default_rng(random_state)

    def select_noise_not_preferred(self):
        """Returns `True` (indicating 'select noise') epsilon of the time."""
        return self._rng.binomial(n=1, p=self.epsilon)

    def __call__(self):
        if self.select_noise_not_preferred():
            return self.noise()
        else:
            return self.preferred()
