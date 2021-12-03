"""Module defines base agent class."""

from abc import ABC, abstractmethod
from typing import final, Any, Callable


class Agent(ABC):
    """Base class for agents."""

    @final
    def action(self, state) -> None:
        """Requests desired action from the agent given state signal.

        By default it calls `_get_action_selector` to retrieve an
        `ActionSelector` instance, which is then called to select a specified
        action.
        """
        # Check that agent is in right state to deliver action
        # (i.e. it's note expecting a reward)
        if hasattr(self, "last_state") or hasattr(self, "last_action"):
            raise RuntimeError(
                "agent hasn't been sent a reward signal for previous action"
            )

        # Select an action to return given the observed state signal.
        action_selector = self._get_action_selector(state)
        chosen_action = action_selector()

        # Save observed state and chosen action for use when processing
        # reward signal to be obtained from environment
        self.last_state = state
        self.last_action = chosen_action

        # Done
        return chosen_action

    @final
    def reward(self, reward) -> None:
        """Sends reward signal `reward` to the agent."""
        # Check that the agent is in right state to process reward signal,
        # i.e. it has selected an action already and hasn't yet received
        # reward.
        if not (hasattr(self, "last_state") or hasattr(self, "last_action")):
            raise RuntimeError(
                "agent's hasn't yet selected an action, so not ready to "
                "process any reward"
            )

        # Call _process_reward method of concrete class
        self._process_reward(self.last_state, self.last_action, reward)

        # Clear last state and action fields to indicate agent is ready
        # to determine its next action.
        del self.last_state, self.last_action

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Returns size of (discrete) action space known to agent."""
        pass

    @abstractmethod
    def _get_action_selector(self, state) -> Callable[[], Any]:
        """Returns (potentially stochastic) `ActionSelector` given state."""
        pass

    @abstractmethod
    def _process_reward(self, last_state, last_action, reward) -> None:
        """Function to be implemented for agent to process received reward."""
        pass
