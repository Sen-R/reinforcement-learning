from unittest.mock import patch
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from rl.policies.action_selection_strategy import EpsilonGreedy, UCB
from rl.action_selector import (
    DeterministicActionSelector,
    UniformDiscreteActionSelector,
    NoisyActionSelector,
)


class TestEpsilonGreedy:
    def test_constructs_correct_object(self) -> None:
        """This selection strategy should create a `NoisyActionSelector`
        instance with correct `epsilon` and other fields. This test checks
        whether that's the case
        """
        epsilon = 0.2
        action_values = [0.0, 1.0, 0.0]
        greedy_action = 1  # argmax(action_values)
        n_actions = 3  # len(action_values)
        random_state = 42
        strategy = EpsilonGreedy(epsilon, random_state=random_state)

        # Test:
        # Test whether strategy, when called with action values vector,
        # returns a correctly configured `NoisyActionSelector` instance
        # corresponding to epsilon greedy action selection.
        s = strategy(action_values)
        assert isinstance(s, NoisyActionSelector)
        assert s.epsilon == epsilon
        assert isinstance(s.preferred, DeterministicActionSelector)
        assert isinstance(s.noise, UniformDiscreteActionSelector)
        assert s.preferred.chosen_action == greedy_action
        assert s.noise.n_actions == n_actions

    def test_rng_sharing(self):
        """Checks that rng is correctly configured and also shared (not
        duplicated) across noise selection and uniform action selection."""
        action_values = [0.0, 0.0, 0.0]  # arbitrary
        epsilon = 0.2  # arbitrary
        random_state = 42  # arbitrary
        strategy = EpsilonGreedy(epsilon, random_state=random_state)
        s = strategy(action_values)

        # Test:
        # If RNG is shared, then we shouldn't get overlapping random numbers
        # when calling the two rng attributes (for NoisyActionSelector
        # and UniformDiscreteActionSelector) sequentially.
        benchmark_rng = np.random.default_rng(random_state)
        assert_array_equal(
            s._rng.random(size=10), benchmark_rng.random(size=10)
        )
        assert_array_equal(
            s.noise._rng.random(size=10), benchmark_rng.random(size=10)
        )

    def test_default_epsilon_is_zero(self):
        strategy = EpsilonGreedy()
        assert strategy.epsilon == 0.0


class TestUCB:
    def test_defaults(self):
        """Tests default arg in __init__."""
        expected_eps = 1e-8
        strategy = UCB(c=2)
        assert strategy._eps == expected_eps

    def test_ucb_method(self):
        """Tests that the UCB method returns `Q(a) + c * sqrt(log(t) / N(a))`
        with epsilon correction where N(a) is 0.
        """
        c = 2
        eps = 1e-8
        action_counts = [3, 5, 0]
        action_values = [1.0, 2.5, 0.0]
        expected_ucbs = np.array(
            [2.66510922e00, 3.78978806e00, 2.88405377e04]
        )  # manual calculation of above formula

        strategy = UCB(c=c, eps=eps)
        ucbs = strategy.ucb(action_values, action_counts)
        assert_array_almost_equal(ucbs / expected_ucbs - 1.0, 0.0)

    def test_call_method(self):
        """Tests whether the strategy's call method correctly uses a
        call to the ucb method to determine which action to return."""
        strategy = UCB(c=2)  # arbitrary c
        ucb_ret_val = [0.0, 1.0, -1.0]
        chosen_action = 1  # argmax(ucb_ret_val)
        with patch.object(
            strategy, "ucb", autospec=True, return_value=ucb_ret_val
        ) as mock_ucb:
            action_selector = strategy("fake_Q", "fake_N")
            mock_ucb.assert_called_with("fake_Q", "fake_N")
            assert isinstance(action_selector, DeterministicActionSelector)
            assert action_selector.chosen_action == chosen_action
