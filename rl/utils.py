"""Utilities module for rl package."""
from typing import Callable, TypeVar, Generic, Optional, Union
from abc import ABC, abstractmethod
from .learningrate import SampleAverageLearningRate, ConstantLearningRate


Param = TypeVar("Param")


def soft_update(current, target, alpha: float):
    """Implements soft updating of `current` towards `target`.

    Implements the following soft-update formula:

    `current := (1 - alpha) * current + alpha * target`

    In the limit `alpha == 0` corresponds to making no change to `current`.
    In the limit `alpha == 1` corresponds to replacing `current` by `target`.
    """
    return (1.0 - alpha) * current + alpha * target


class UpdatableParameter(ABC, Generic[Param]):
    """ABC for parameters that update (typically using soft-update rule) based
    on observations."""

    @abstractmethod
    def __call__(self) -> Param:
        """Returns the current value of the parameter."""
        pass

    @abstractmethod
    def update(self, value: Param) -> None:
        """Uses new observation `value` to update the parameter value."""
        pass


class SoftUpdatedParameter(UpdatableParameter):
    """Object to hold parameter and functionality to soft update it.

    Args:
      initial_value: initial value of the parameter
      learning_rate_schedule: callable determining alpha(n)
    """

    def __init__(
        self,
        initial_value: Param,
        learning_rate_schedule: Optional[
            Union[float, Callable[[int], float]]
        ] = None,
    ):
        self._value = initial_value
        if callable(learning_rate_schedule):
            self._learning_rate_schedule = learning_rate_schedule
        elif learning_rate_schedule is None:
            self._learning_rate_schedule = SampleAverageLearningRate()
        else:
            self._learning_rate_schedule = ConstantLearningRate(
                float(learning_rate_schedule)
            )
        self._n = 0

    def __call__(self) -> Param:
        return self._value

    def update(self, value: Param) -> None:
        self._value = soft_update(
            self._value, value, self._learning_rate_schedule(self._n)
        )
        self._n += 1


class FixedParameter(UpdatableParameter):
    """Special case of SoftUpdatedParameter with alpha=0.

    I.e. the parameter remains its initial value even after calling
    `update` method.
    """

    def __init__(self, initial_value: Param):
        self._value = initial_value

    def __call__(self) -> Param:
        return self._value

    def update(self, value: Param) -> None:
        # No-op for FixedParameter
        pass
