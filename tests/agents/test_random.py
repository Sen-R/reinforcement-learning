import unittest
import numpy as np
from rl.agents.random import DiscreteRandomAgent
from rl.agents.action_selector import UniformDiscreteActionSelector


class TestRandomAgent(unittest.TestCase):
    def test_agent_action(self) -> None:
        """Tests whether agent can return an action."""
        agent = DiscreteRandomAgent(5, random_state=32)
        action = agent.action(state=None)
        self.assertEqual(action, 4)

    def test_agent_returns_uniform_discrete_action_selector(self) -> None:
        k = 5  # arbitrary
        agent = DiscreteRandomAgent(k)
        action_selector = agent._get_action_selector(state=None)
        assert isinstance(action_selector, UniformDiscreteActionSelector)
        self.assertEqual(action_selector.n_actions, k)

    def test_agent_action_works_with_different_state_signals(self) -> None:
        """Tests whether agent method works with variety of state signals."""
        for state in [None, 3, [1, 2], np.array([[1.0, 2.0], [3.0, 4.0]])]:
            with self.subTest(state=state):
                agent = DiscreteRandomAgent(5)
                action = agent.action(state)
                self.assertGreaterEqual(action, 0)

    def test_agent_reward(self) -> None:
        """Tests whether agent has reward method.

        Note that for a random agent, this doesn't do anything, but method
        should be provided for compatibility with other agents.
        """
        agent = DiscreteRandomAgent(5, random_state=32)
        agent.action(state=None)
        agent.reward(3)
