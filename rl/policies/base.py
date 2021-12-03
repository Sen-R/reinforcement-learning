"""Abstract base classes for policies subpackage."""

from abc import ABC, abstractmethod
from rl.action_selector import ActionSelector


class Policy(ABC):
    """Base class for all policies.

    Semantically, represents a mapping from states to distributions
    over the action space (represented by `ActionSelector` objects).
    """

    @abstractmethod
    def __call__(self, state) -> ActionSelector:
        """Returns (possibly stochastic) `ActionSelector` for given state."""
        pass


class LearningPolicy(Policy):
    """Base class for learning policies.

    These policies have an `update` method that allows the policy to update
    itself (according to the learning logic implemented in the concrete class)
    in response to observed state-action-reward triples.
    """

    @abstractmethod
    def update(self, state, action, reward) -> None:
        """Updates policy given state-action-reward triple of experience."""
        pass
