"""Module implementing non-associative agents.

These are agents that ignore state signals. Useful for learning in
nonassociative settings, e.g. multi-armed bandits.
"""

from typing import List, Optional, Sequence
import numpy as np
from .base import Agent
from ..custom_types import LearningRateSchedule
from .action_selector import EpsilonGreedyActionSelector, NoisyActionSelector
from ..learningrate import SampleAverageLearningRate
from ..utils import soft_update


class RewardAveragingEpsilonGreedyAgent(Agent):
    """Implementation of epsilon greedy short-termist agent.

    This agent is both epsilon greedy in how it selects actions and
    short-termist in the sense that the perceived value of an action
    is equated to the immediate reward obtained after the action. This is
    fine for environments where consecutive actions yield independent
    rewards, such as standard multi-armed bandit problems.

    Agent learns by soft updating its estimated action values by the
    reward that immediately follows each action. The alpha parameter for
    the soft update rule can be controlled as a function of the number of
    times that action has been taken in the past, allowing e.g. for
    unweighted averaging over all observed rewards (per action) or for
    exponential weighting towards recent observations.

    Args:
      n_actions: size of the action space
      learning_rate_schedule: function mapping action count to soft update
        parameter alpha. Can be one of the objects provided in the module
        `rl.learningrate`
      epsilon: probability of taking an action to explore rather than exploit
      initial_action_values: initial estimates of the value of each action, by
        default set to zero for all actions
      random_state: `None`, `int`, `np.random.Generator` etc to initialise RNG
    """

    def __init__(
        self,
        n_actions: int,
        *,
        learning_rate_schedule: LearningRateSchedule = None,
        epsilon: float = 0.0,
        initial_action_values: Optional[Sequence[float]] = None,
        random_state=None,
    ) -> None:
        self._n_actions = n_actions
        self._action_counts = [0] * n_actions
        self.epsilon = epsilon
        if initial_action_values is None:
            initial_action_values = [0.0] * n_actions
        else:
            initial_action_values = list(initial_action_values)
        self._action_values = initial_action_values
        self._rng = np.random.default_rng(random_state)
        if learning_rate_schedule is None:
            learning_rate_schedule = SampleAverageLearningRate()
        self.alpha = learning_rate_schedule

    @property
    def action_counts(self) -> List[int]:
        return self._action_counts

    @property
    def Q(self) -> List[float]:
        """Returns array of estimated action values."""
        return self._action_values

    def _process_reward(
        self, last_state, last_action: int, reward: float
    ) -> None:
        # Update action values first
        n = self._action_counts[last_action]
        alpha_n = self.alpha(n)
        self._action_values[last_action] = soft_update(
            self._action_values[last_action], reward, alpha_n
        )

        # Update action counts after
        # TODO: refactor into base class?
        self._action_counts[last_action] += 1

    def _get_action_selector(self, state=None) -> NoisyActionSelector:
        desired_action = int(np.argmax(self.Q))
        return EpsilonGreedyActionSelector(
            self.epsilon,
            desired_action,
            self._n_actions,
            random_state=self._rng,
        )
