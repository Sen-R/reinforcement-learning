from typing import Dict, List, Any
import rl.simulator as simulator


class Callback:
    """Base class for implementing custom callbacks."""

    def __call__(
        self,
        sim: "simulator.SingleAgentWaitingSimulator",
        state,
        action,
        reward,
        done: bool,
    ) -> None:
        """This method is called at the end of each step."""
        pass


class History(Callback):
    """Class for keeping track of simulation history."""

    def __init__(self, logging_period: int = 1):
        self.states: List = []
        self.actions: List = []
        self.rewards: List = []
        self.logging_period = logging_period

    def __call__(
        self,
        sim: "simulator.SingleAgentWaitingSimulator",
        state,
        action,
        reward,
        done: bool,
    ):
        """Adds state-action-reward triple to history."""
        if (sim.t % self.logging_period) == 0:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)

    def to_dict(self) -> Dict[str, List]:
        """Exports history as python dictionary."""
        return {
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
        }


class AgentStateLogger(Callback):
    """Periodically logs the state of the agent.

    Args:
      logging_period: logs the state every `logging_period` steps
    """

    def __init__(self, logging_period: int = 1):
        self.logging_period = logging_period
        self._states: List[Dict[str, Any]] = []

    def __call__(
        self,
        sim: "simulator.SingleAgentWaitingSimulator",
        state,
        action,
        reward,
        done,
    ) -> None:
        if (sim.t % self.logging_period) == 0:
            self._states.append(sim.agent.state)

    @property
    def states(self) -> List[Dict[str, Any]]:
        return self._states


class EnvironmentStateLogger(Callback):
    """Periodically logs the (full) state of the environment.

    Args:
      logging_period: logs the state every `logging_period` steps
    """

    def __init__(self, logging_period: int = 1):
        self.logging_period = logging_period
        self._states: List = []

    def __call__(
        self,
        sim: "simulator.SingleAgentWaitingSimulator",
        state,
        action,
        reward,
        done,
    ):
        if (sim.t % self.logging_period) == 0:
            self._states.append(sim.environment.state)

    @property
    def states(self) -> List:
        return self._states
