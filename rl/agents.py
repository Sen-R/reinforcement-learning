from typing import Optional, Sequence
from .agent import Agent
from .custom_types import LearningRateSchedule
from .policies.value_learning import RewardAveragingPolicy
from .policies.action_selection_strategy import EpsilonGreedy, UCB


class EpsilonGreedyRewardAveragingAgent(Agent):
    """Epsilon-greedy agent that estimates action values by averaging
    immediate rewards.

    Args:
      epsilon: probability of taking random exploration action
      n_actions: size of action space
      learning_rate_schedule: function mapping action count to soft update
        parameter `alpha`. Can be one of the objects provided in
        `rl.learningrate`
      initial_action_values: initial estimates of the value of each action,
        by default set to zero for all actions
      random_state: `None`, `int` or `np.random.Generator` to initialise RNG
        used to select random actions (if exploring)
    """

    def __init__(
        self,
        epsilon: float,
        n_actions: int,
        *,
        learning_rate_schedule: LearningRateSchedule = None,
        initial_action_values: Optional[Sequence[float]] = None,
        random_state=None,
    ):
        policy = RewardAveragingPolicy(
            n_actions,
            action_selection_strategy=EpsilonGreedy(epsilon, random_state),
            learning_rate_schedule=learning_rate_schedule,
            initial_action_values=initial_action_values,
        )
        super().__init__(policy)


class UCBRewardAveragingAgent(Agent):
    """Agent that estimates values by averaging immediate rewards and selects
    actions using Upper Confidence Bound selection.

    See Chapter 2 of Sutton, Barto (2018) for details.

    Args:
      c: `c` parameter for Upper Confidence Bound method
      n_actions: size of action space
      learning_rate_schedule: function mapping action count to soft update
        parameter `alpha`. Can be one of the objects provided in
        `rl.learningrate`
      initial_action_values: initial estimates of the value of each action,
        by default set to zero for all actions
    """

    def __init__(
        self,
        c: float,
        n_actions: int,
        *,
        learning_rate_schedule: LearningRateSchedule = None,
        initial_action_values: Optional[Sequence[float]] = None,
    ):
        policy = RewardAveragingPolicy(
            n_actions,
            action_selection_strategy=UCB(c),
            learning_rate_schedule=learning_rate_schedule,
            initial_action_values=initial_action_values,
        )
        super().__init__(policy)
