"""Implementation of agents that take random actions."""

from typing import Final
from .base import Agent
from .action_selector import UniformDiscreteActionSelector


class DiscreteRandomAgent(Agent):
    """Randomly acting agent for discrete action spaces."""

    def __init__(self, n_actions: int, random_state=None) -> None:
        self._n_actions: int = n_actions
        self._action_selector: Final = UniformDiscreteActionSelector(
            self._n_actions, random_state=random_state
        )

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def _get_action_selector(self, _) -> UniformDiscreteActionSelector:
        return self._action_selector

    def _process_reward(self, last_state, last_action, reward) -> None:
        """No-op in the case of a random agent."""
        pass
