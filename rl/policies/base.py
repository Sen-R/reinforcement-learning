"""Abstract base classes for policies subpackage."""

from abc import ABC, abstractmethod
from rl.action_selectors import ActionSelector


class Policy(ABC):
    """Base class for all policies.

    Semantically, represents a mapping from states to distributions
    over the action space (represented by `ActionSelector` objects).

    In general, policies have an `update` method to allow it to learn
    from state-action-reward experiences.
    """

    @abstractmethod
    def __call__(self, state) -> ActionSelector:
        """Returns (possibly stochastic) `ActionSelector` for given state."""
        pass

    @abstractmethod
    def update(self, state, action, reward) -> None:
        """Updates policy given state-action-reward triple of experience."""
        pass


class DumbPolicy(Policy):
    """Base class for policies that do not learn.

    For such policies, the `update` method is a no-op.
    """

    def update(self, state, action, reward) -> None:
        pass
