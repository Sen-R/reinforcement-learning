"""Module implementing non-associative agents.

These are agents that ignore state signals. Useful for learning in
nonassociative settings, e.g. multi-armed bandits.
"""

from typing import List, Optional, Sequence, Dict, Any
from .base import Policy
from ..action_selectors import ActionSelector
from .action_selection_strategy import ActionSelectionStrategy
from ..custom_types import LearningRateSchedule
from ..learningrate import SampleAverageLearningRate
from ..utils import soft_update


class RewardAveragingPolicy(Policy):
    """Implementation of short-termist policy.

    This policy is short-termist in the sense that the perceived value of
    an action
    is equated to the immediate reward obtained after the action. This is
    fine for environments where consecutive actions yield independent
    rewards, such as standard multi-armed bandit problems.

    Policy learns by soft updating its estimated action values by the
    reward that immediately follows each action. The alpha parameter for
    the soft update rule can be controlled as a function of the number of
    times that action has been taken in the past, allowing e.g. for
    unweighted averaging over all observed rewards (per action) or for
    exponential weighting towards recent observations.

    Args:
      n_actions: size of the action space
      action_selection_strategy: `ActionSelectionStrategy` instance, e.g.
        `EpsilonGreedy`
      learning_rate_schedule: function mapping action count to soft update
        parameter alpha. Can be one of the objects provided in the module
        `rl.learningrate`
      initial_action_values: initial estimates of the value of each action, by
        default set to zero for all actions
    """

    def __init__(
        self,
        n_actions: int,
        *,
        action_selection_strategy: ActionSelectionStrategy,
        learning_rate_schedule: LearningRateSchedule = None,
        initial_action_values: Optional[Sequence[float]] = None,
    ) -> None:
        self.action_selection_strategy = action_selection_strategy
        self._action_counts = [0] * n_actions
        if initial_action_values is None:
            initial_action_values = [0.0] * n_actions
        else:
            initial_action_values = list(initial_action_values)
        self._action_values = initial_action_values
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

    def update(self, last_state, last_action: int, reward: float) -> None:
        # Update action values first
        n = self._action_counts[last_action]
        alpha_n = self.alpha(n)
        self._action_values[last_action] = soft_update(
            self._action_values[last_action], reward, alpha_n
        )

        # Update action counts after
        # TODO: refactor into base class?
        self._action_counts[last_action] += 1

    def __call__(self, state=None) -> ActionSelector:
        return self.action_selection_strategy(self.Q, self.action_counts)

    @property
    def state(self) -> Dict[str, Any]:
        return {"Q": self.Q.copy(), "action_counts": self.action_counts.copy()}
