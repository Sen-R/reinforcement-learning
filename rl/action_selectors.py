"""Action selector implementations.

Action selectors are objects that when called return a desired
action. These actions may be stochastically chosen (e.g. randomly chosen
from a list of candidates) depending on the choice of `ActionSelector`
implementation, and how it is configured.

Examples include the following
* `DeterministicActionSelector`: always returns the same (specified) action
* `UniformDiscreteActionSelector`: selections an action uniformly at random
  from a specified discrete action space
* `NoisyActionSelector`: uses either a "preferred" action selector (with
  probability `1 - epsilon`) or a "noise" action selector (with probability
  `epsilon`) to determine the action. Useful, for example, to implement an
  epsilon-greedy agent.
"""

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

    def __init__(self, n_actions: int, *, random_state=None):
        self.n_actions = n_actions
        self._rng = np.random.default_rng(random_state)

    def __call__(self) -> int:
        return self._rng.integers(self.n_actions)


class NoisyActionSelector(ActionSelector):
    """Noisy action selector.

    With probability `1 - epsilon` this uses `preferred_selector` to
    select actions; with probability `epsilon` this uses `noise_selector`
    to select actions. Useful, for example, for implementing an
    epsilon-greedy agent, or any other agent (e.g. continuous action spaces)
    where you wish to inject noise into action decisions.
    """

    def __init__(
        self,
        epsilon: float,
        preferred_selector: ActionSelector,
        noise_selector: ActionSelector,
        *,
        random_state=None,
    ):
        self.epsilon = epsilon
        self.preferred = preferred_selector
        self.noise = noise_selector
        self._rng = np.random.default_rng(random_state)

    def select_noise_not_preferred(self) -> bool:
        """Returns `True` (indicating 'select noise') epsilon of the time."""
        return bool(self._rng.binomial(n=1, p=self.epsilon))

    def __call__(self):
        if self.select_noise_not_preferred():
            return self.noise()
        else:
            return self.preferred()


class EpsilonGreedyActionSelector(NoisyActionSelector):
    """Specialised `NoisyActionSelector` for epsilon-greedy selection.

    Subclass of `NoisyActionSelector` configured for epsilon greedy action
    selection from a discrete action space.

    Args:
      epsilon: probability of choosing a (uniformly) random action
      chosen_action: desired (greedy) action
      n_actions: size of discrete action space

    Returns:
      `NoisyActionSelector` instance that when called performs epsilon-greedy
      action selection.
    """

    # Specialising superclass types for this subclass (so type checker knows
    # their specialised types)
    preferred: DeterministicActionSelector
    noise: UniformDiscreteActionSelector

    def __init__(
        self,
        epsilon: float,
        chosen_action: int,
        n_actions: int,
        *,
        random_state=None,
    ):
        rng = np.random.default_rng(random_state)
        preferred = DeterministicActionSelector(chosen_action)
        noise = UniformDiscreteActionSelector(n_actions, random_state=rng)
        super().__init__(epsilon, preferred, noise, random_state=rng)
