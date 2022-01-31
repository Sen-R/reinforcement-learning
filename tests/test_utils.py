from rl.utils import (
    soft_update,
    SoftUpdatedParameter,
    FixedParameter,
    UpdatableParameter,
)
from rl.learningrate import SampleAverageLearningRate


class TestSoftUpdate:
    def test_function_is_correct(self) -> None:
        alpha = 0.2
        current = 1.0
        target = 1.5
        expected = 1.1  # (1 - alpha) * current + alpha * target
        assert soft_update(current, target, alpha) == expected


class TestSoftUpdatedParameter:
    def test_functionality(self) -> None:
        # Set up parameter
        lr_schedule = SampleAverageLearningRate()
        param = SoftUpdatedParameter(
            initial_value=1.0, learning_rate_schedule=lr_schedule
        )

        # Test param is a subclass of UpdatableParameter
        assert isinstance(param, UpdatableParameter)

        # Test initial value is correct
        assert param() == 1.0

        # Test value is correctly updated after first iteration
        param.update(2.0)
        assert param() == 2.0

        # Test value is correctly updated after second iteration (and
        # implicitly that alpha has been updated to facilitate this)
        param.update(3.0)
        assert param() == 2.5

    def test_providing_number_as_learning_rate(self) -> None:
        """When a number is provided as the learning rate, this should
        have been converted into a ConstantLearningRate schedule."""
        param = SoftUpdatedParameter(0.0, 0.1)
        param.update(1.0)
        assert param() == 0.1
        param.update(0.5)
        assert param() == 0.1 * 0.5 + 0.9 * 0.1

    def test_providing_default_learning_rate_schedule_is_sample_averaging(
        self,
    ) -> None:
        param = SoftUpdatedParameter(0.0)
        param.update(1.0)
        assert param() == 1.0
        param.update(-1.0)
        assert param() == 0.0
        param.update(3.0)
        assert param() == 1.0


class TestFixedParameter:
    def test_functionality(self):
        param = FixedParameter(1.0)

        # Should be a subclass of UpdatableParameter
        assert isinstance(param, UpdatableParameter)

        # Should return the initial value no matter whether this has
        # been "updated"
        assert param() == 1.0
        param.update(2.0)
        assert param() == 1.0
        param.update(3.0)
        assert param() == 1.0
