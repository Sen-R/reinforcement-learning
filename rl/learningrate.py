"""Collection of commonly used learning rate schedule functions."""


from abc import ABC, abstractmethod


class LearningRateSchedule(ABC):
    @abstractmethod
    def __call__(self, current_step_number):
        """Returns learning rate for `step_number`."""
        pass


class ConstantLearningRate(LearningRateSchedule):
    """Constant learning rate schedule of `alpha`."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, _):
        return self.alpha


class SampleAverageLearningRate(LearningRateSchedule):
    """Decaying "sample average" learning rate schedule.

    A learning rate schedule of the form `alpha_n = 1 / (n + 1)`.

    Called "sample average" because when it is used in a soft-update
    rule with initial value set to zero and step numbering starting at
    zero, the resulting estimates track the (unweighted) average of
    all samples observed.
    """

    def __call__(self, n):
        return 1.0 / (n + 1)
