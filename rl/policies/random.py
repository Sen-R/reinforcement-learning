"""Implementation of agents that take random actions."""

from typing import Final
from .base import DumbPolicy
from ..action_selector import UniformDiscreteActionSelector


class DiscreteRandomPolicy(DumbPolicy):
    """Policy to take uniformly random action in discrete action space."""

    def __init__(self, n_actions: int, *, random_state=None):
        self._action_selector: Final = UniformDiscreteActionSelector(
            n_actions, random_state=random_state
        )

    def __call__(self, state) -> UniformDiscreteActionSelector:
        return self._action_selector
