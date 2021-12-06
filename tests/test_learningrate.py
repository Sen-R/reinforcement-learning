import pytest
from rl.learningrate import (
    ConstantLearningRate,
    SampleAverageLearningRate,
)


class TestConstantLearningRate:
    @pytest.mark.parametrize("n", [0, 1, 10])
    def test_function_is_correct(self, n) -> None:
        alpha = 0.01
        lr = ConstantLearningRate(alpha)
        assert lr(n) == alpha


class TestSampleAverageLearningRate:
    @pytest.mark.parametrize("n", [0, 1, 10])
    def test_function_is_correct(self, n) -> None:
        lr = SampleAverageLearningRate()
        assert lr(n) == 1.0 / (n + 1)
