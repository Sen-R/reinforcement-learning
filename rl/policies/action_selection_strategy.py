"""Strategies for selecting actions for value-based policies."""

from abc import ABC, abstractmethod
from typing import List, Optional
from numpy.typing import ArrayLike
import numpy as np
from rl.action_selectors import (
    ActionSelector,
    DeterministicActionSelector,
    UniformDiscreteActionSelector,
    NoisyActionSelector,
)


class ActionSelectionStrategy(ABC):
    """Base class for action selection strategies."""

    @abstractmethod
    def __call__(
        self,
        action_values: List[float],
        action_counts: List[int],
    ) -> ActionSelector:
        pass


class EpsilonGreedy(ActionSelectionStrategy):
    """Implementation of epsilon greedy action selection.

    Args:
      epsilon: probability of taking action to explore rather than exploing
      random_state: `None`, `int`, or `np.random.Generator` to initialise
        RNG
    """

    def __init__(self, epsilon: float = 0.0, random_state=None):
        self.epsilon = epsilon
        self._rng = np.random.default_rng(random_state)

    def __call__(
        self,
        action_values: List[float],
        action_counts: Optional[List[int]] = None,
    ) -> NoisyActionSelector:
        """Action counts do not matter for this strategy."""
        greedy_action = int(np.argmax(action_values))
        preferred = DeterministicActionSelector(greedy_action)
        noise = UniformDiscreteActionSelector(
            len(action_values), random_state=self._rng
        )
        return NoisyActionSelector(
            self.epsilon, preferred, noise, random_state=self._rng
        )


class UCB(ActionSelectionStrategy):
    """Upper confidence bound action selection strategy.

    As defined in Sutton & Barto equation 2.10. However we floor action
    counts at `eps` to avoid divide-by-zero.

    `t` is inferred by summing the action counts vector and adding 1.
    (Because `t` refers to the time step at which action values are being
    estimated, i.e. the next time step since the last observation).

    Args:
      c: confidence parameter
      eps: small number to floor zero counts at
    """

    def __init__(self, c: float, eps: float = 1.0e-8):
        self.c = c
        self._eps = eps

    def __call__(
        self,
        action_values: List[float],
        action_counts: List[int],
    ) -> DeterministicActionSelector:
        chosen_action = int(np.argmax(self.ucb(action_values, action_counts)))
        return DeterministicActionSelector(chosen_action)

    def ucb(
        self,
        action_values: List[float],
        action_counts: List[int],
    ) -> ArrayLike:
        log_t = np.log(np.sum(action_counts) + 1)
        floored_counts = np.maximum(action_counts, self._eps)
        return action_values + self.c * np.sqrt(log_t / floored_counts)
