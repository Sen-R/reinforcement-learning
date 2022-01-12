"""Module implementing simulation engine."""
from typing import Dict, List, Iterable, Callable, Optional
from rl.environments.base import Environment
from rl.agent import Agent


class History:
    """Class for keeping track of simulation history."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        """Adds state-action-reward triple to history."""
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


class SingleAgentWaitingSimulator:
    """Simulator for single agent turn-based problems.

    This simulator is suitable for discrete time-step problems, i.e.
    where the environment waits for the agent to decide on an action.
    """

    def __init__(
        self,
        environment: Environment,
        agent: Agent,
        callbacks: Optional[
            Iterable[Callable[["SingleAgentWaitingSimulator"], None]]
        ] = None,
    ):
        self.environment = environment
        self.agent = agent
        self.history = History()
        self.callbacks = [] if callbacks is None else list(callbacks)

    @property
    def t(self) -> int:
        """Returns the current time step."""
        return len(self.history.states)

    def run(self, n_steps: int):
        """Runs the simulation for `n_steps` time steps.

        The outcome of the simulation is recorded in `self.history`.
        """
        for _ in range(n_steps):
            state = self.environment.observe()
            action = self.agent.action(state)
            reward = self.environment.act(action)
            self.agent.reward(reward)
            self.history.add(state, action, reward)
            for callback in self.callbacks:
                callback(self)
            if self.environment.done:
                break
