import unittest
from unittest.mock import Mock, patch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_less
from rl.agents.action_selector import (
    ActionSelector,
    DeterministicActionSelector,
    UniformDiscreteActionSelector,
    NoisyActionSelector,
)


def binom_mean(n, p):
    return n * p


def binom_stddev(n, p):
    return np.sqrt(n * p * (1.0 - p))


class TestDeterministicActionSelector(unittest.TestCase):
    def test_calling_always_return_the_chosen_action(self):
        chosen_action = 0
        n_trials = 10
        s = DeterministicActionSelector(chosen_action)
        for _ in range(n_trials):
            self.assertEqual(s(), chosen_action)


class TestUniformDiscreteActionSelector(unittest.TestCase):
    def test_calling_returns_uniformly_sampled_actions(self):
        """Note that this test is statistical and could fail due to
        bad luck (Type I error), or could pass erroneously (Type II error).

        Tune the `margin` variable below to balance Type I and Type II errors.

        Alternatively you can increase `n_trials` to make the test more
        reliable, at the expense of computational time.

        Once the test passes with a small margin, it should stay that way
        unless the underlying RNG implementation is changed.
        """
        # Test parameters
        # TODO: refactor into test config file?
        n_actions = 3
        n_trials = 1000
        expected_counts = binom_mean(n_trials, 1.0 / n_actions)
        stddev = binom_stddev(n_trials, 1.0 / n_actions)
        margin = 0.25 * stddev  # <-- tune if necessary following regression

        # Run the test
        s = UniformDiscreteActionSelector(n_actions, random_state=42)
        sampled_actions = [s() for _ in range(n_trials)]
        action_values, action_counts = np.unique(
            sampled_actions, return_counts=True
        )

        assert_array_equal(action_values, range(n_actions))
        assert_array_less(expected_counts - margin, action_counts)
        assert_array_less(action_counts, expected_counts + margin)


class TestNoisyActionSelector(unittest.TestCase):
    def test_desired_or_noise_method(self):
        """Tests `select_noise_not_preferred` method.

        This method should return True a fraction of the time as specified
        by `epsilon` when the selector was initialised.

        NB this is another statistical test, usual caveats (see above) apply.
        """
        # Test parameters
        # TODO: refactor into config file?
        n_actions = 3
        preferred_action = 0
        epsilon = 0.2
        n_trials = 1000
        expected_count = binom_mean(n_trials, epsilon)
        expected_stddev = binom_stddev(n_trials, epsilon)
        margin = 0.25 * expected_stddev  # <-- tune if necessary

        # Run test
        s = NoisyActionSelector(
            epsilon,
            DeterministicActionSelector(preferred_action),
            UniformDiscreteActionSelector(n_actions),
            random_state=42,
        )
        samples = [s.select_noise_not_preferred() for _ in range(n_trials)]
        values, counts = np.unique(samples, return_counts=True)
        assert_array_equal(values, [False, True])
        self.assertLess(expected_count - margin, counts[1])
        self.assertLess(counts[1], expected_count + margin)

    def test_call(self):
        """Tests whether call:
        1. calls `select_noise_not_preferred` to determine how to select
           action.
        2. returns preferred when `select_noise_not_preferred` is false
        3. returns noise otherwise
        """
        epsilon = 1
        preferred = Mock(spec=ActionSelector)
        noise = Mock(spec=ActionSelector)
        with patch.object(
            NoisyActionSelector,
            "select_noise_not_preferred",
            side_effect=[False, True],
        ):
            s = NoisyActionSelector(epsilon, preferred, noise)
            # First time should call preferred
            s()
            preferred.assert_called_with()
            noise.assert_not_called()
            # Reset mocks
            preferred.reset_mock()
            noise.reset_mock()
            # Second time should call noise
            s()
            preferred.assert_not_called()
            noise.assert_called_with()
