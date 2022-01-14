from typing import Dict, List, Any
from rl.simulator import SingleAgentWaitingSimulator


class Callback:
    """Base class for implementing custom callbacks."""

    def __call__(self, sim: SingleAgentWaitingSimulator) -> None:
        """This method is called at the end of each step."""
        pass


class AgentStateLogger(Callback):
    """Periodically logs the state of the agent.

    Args:
      logging_period: logs the state every `logging_period` steps
    """

    def __init__(self, logging_period: int = 1):
        self.logging_period = logging_period
        self._states: List[Dict[str, Any]] = []

    def __call__(self, sim: SingleAgentWaitingSimulator) -> None:
        if (sim.t % self.logging_period) == 0:
            self._states.append(sim.agent.state)

    @property
    def states(self) -> List[Dict[str, Any]]:
        return self._states
