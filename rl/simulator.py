"""Module implementing simulation engine."""
from typing import Iterable, Optional
from rl.environments.base import Environment
from rl.agent import Agent
import rl.callbacks as callbacks


class SingleAgentWaitingSimulator:
    """Simulator for single agent turn-based problems.

    This simulator is suitable for discrete time-step problems, i.e.
    where the environment waits for the agent to decide on an action.
    """

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        callbacks: Optional[Iterable["callbacks.Callback"]] = None,
    ):
        self.environment = environment
        self.agent = agent
        self.callbacks = [] if callbacks is None else list(callbacks)
        self._t = 0

    @property
    def t(self) -> int:
        """Returns the current time step."""
        return self._t

    def run(self, n_steps: int):
        """Runs the simulation for `n_steps` time steps.

        The outcome of the simulation is recorded in `self.history`.
        """
        for _ in range(n_steps):
            state = self.environment.observe()
            action = self.agent.action(state)
            reward = self.environment.act(action)
            self.agent.reward(reward)
            done = self.environment.done
            self._t += 1
            for callback in self.callbacks:
                callback(self, state, action, reward, done)
            if done:
                break
