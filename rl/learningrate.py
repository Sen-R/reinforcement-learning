"""Collection of commonly used learning rate schedule functions."""


class ConstantLearningRate:
    """Constant learning rate schedule of `alpha`."""

    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, _):
        return self.alpha


class SampleAverageLearningRate:
    """Decaying "sample average" learning rate schedule.

    A learning rate schedule of the form `alpha_n = 1 / (n + 1)`.

    Called "sample average" because when it is used in a soft-update
    rule with initial value set to zero and step numbering starting at
    zero, the resulting estimates track the (unweighted) average of
    all samples observed.
    """

    def __call__(self, n):
        return 1.0 / (n + 1)
