import numpy as np
from .base import Agent
from .action_selector import EpsilonGreedyActionSelector
from ..learningrate import SampleAverageLearningRate
from ..utils import soft_update


class RewardAveragingEpsilonGreedyAgent(Agent):
    def __init__(
        self,
        n_actions,
        *,
        learning_rate_schedule=None,
        epsilon=0.0,
        initial_action_values=0.0,
        random_state=None,
    ):
        self._n_actions = n_actions
        self._action_counts = np.zeros(n_actions)
        self.epsilon = epsilon
        self._action_values = initial_action_values * np.ones(n_actions)
        self._rng = np.random.default_rng(random_state)
        if learning_rate_schedule is None:
            learning_rate_schedule = SampleAverageLearningRate()
        self.alpha = learning_rate_schedule

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def action_counts(self):
        return self._action_counts

    @property
    def Q(self):
        """Returns array of estimated action values."""
        return self._action_values

    def _process_reward(self, last_state, last_action, reward):
        # Update action values first
        n = self._action_counts[last_action]
        alpha_n = self.alpha(n)
        self._action_values[last_action] = soft_update(
            self._action_values[last_action], reward, alpha_n
        )

        # Update action counts after
        # TODO: refactor into base class?
        self._action_counts[last_action] += 1

    def _get_action_selector(self, state=None):
        desired_action = np.argmax(self.Q)
        return EpsilonGreedyActionSelector(
            self.epsilon,
            desired_action,
            self._n_actions,
            random_state=self._rng,
        )
