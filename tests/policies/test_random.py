import unittest
import numpy as np
from rl.policies.random import DiscreteRandomPolicy
from rl.action_selector import UniformDiscreteActionSelector


class TestDiscreteRandomPolicy(unittest.TestCase):
    def test_policy_returns_uniform_discrete_action_selector(self) -> None:
        k = 5  # arbitrary
        policy = DiscreteRandomPolicy(k)
        action_selector = policy(state=None)
        assert isinstance(action_selector, UniformDiscreteActionSelector)
        self.assertEqual(action_selector.n_actions, k)

    def test_policy_works_with_different_state_signals(self) -> None:
        """Tests whether policy's call method works with variety of state
        signals."""
        for state in [None, 3, [1, 2], np.array([[1.0, 2.0], [3.0, 4.0]])]:
            with self.subTest(state=state):
                policy = DiscreteRandomPolicy(5)
                action = policy(state)
                self.assertIsInstance(action, UniformDiscreteActionSelector)
