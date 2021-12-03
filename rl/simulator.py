"""Module implementing simulation engine."""
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


class SingleAgentWaitingSimulator:
    """Simulator for single agent turn-based problems.

    This simulator is suitable for discrete time-step problems, i.e.
    where the environment waits for the agent to decide on an action.
    """

    def __init__(self, environment: Environment, agent: Agent):
        self.environment = environment
        self.agent = agent
        self.history = History()

    @property
    def t(self) -> int:
        """Returns the current time step."""
        return len(self.history.states)

    def run(self, n_steps: int):
        """Runs the simulation for `n_steps` time steps.

        The outcome of the simulation is recorded in `self.history`.
        """
        for _ in range(n_steps):
            state = self.environment.state()
            action = self.agent.action(state)
            reward = self.environment.act(action)
            self.agent.reward(reward)
            self.history.add(state, action, reward)
