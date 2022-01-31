from typing import Dict, Any, Optional, Iterable, Union
import numpy as np
from .base import Policy
from ..action_selectors import DiscreteActionSelector
from ..utils import SoftUpdatedParameter, UpdatableParameter, FixedParameter


class GradientBandit(Policy):
    """Implementation of Gradient Bandit policy.

    See chapter 2 of Sutton, Barto (2018) for further details.

    Args:
      alpha: learning rate for gradient ascent
      n_actions: size of the action space
      baseline: object of type `UpdatableParameter` or constant float to
        use as the baseline. If `None`, defaults to sample averaged historical
        rewards
      initial_preferences: initial values to set `H` vector
      random_state: initial state for RNG
    """

    def __init__(
        self,
        alpha: float,
        n_actions: int,
        *,
        baseline: Optional[Union[float, UpdatableParameter]] = None,
        initial_preferences: Optional[Iterable[float]] = None,
        random_state=None,
    ):
        self.alpha = alpha
        if isinstance(baseline, UpdatableParameter):
            self._baseline = baseline
        elif baseline is None:
            self._baseline = SoftUpdatedParameter(0.0)
        else:
            self._baseline = FixedParameter(baseline)

        if initial_preferences is None:
            self.H = np.zeros(n_actions)
        else:
            self.H = np.array(initial_preferences)

        if len(self.H) != n_actions:
            raise ValueError("incorrect length for initial_preferences")

        self._rng = np.random.default_rng(random_state)

    @property
    def baseline(self) -> float:
        return self._baseline()

    def __call__(self, state) -> DiscreteActionSelector:
        return DiscreteActionSelector(self.p, random_state=self._rng)

    def update(self, state, action, reward) -> None:
        self._baseline.update(reward)
        reward_minus_baseline = reward - self.baseline
        self.H += -self.alpha * reward_minus_baseline * self.p
        self.H[action] += self.alpha * reward_minus_baseline

    @property
    def state(self) -> Dict[str, Any]:
        return {"H": self.H.copy()}

    @property
    def p(self) -> np.ndarray:
        """Softmax implementation."""
        exp_H = np.exp(self.H)
        return exp_H / np.sum(exp_H)
