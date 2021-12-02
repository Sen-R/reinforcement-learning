import unittest
import numpy as np
from rl.agents.random import DiscreteRandomAgent


class TestRandomAgent(unittest.TestCase):
    def test_agent_properties(self):
        n_actions = 5
        agent = DiscreteRandomAgent(n_actions)
        self.assertEqual(agent.n_actions, n_actions)

    def test_agent_action(self):
        """Tests whether agent can return an action."""
        agent = DiscreteRandomAgent(5, random_state=32)
        action = agent.action()
        self.assertEqual(action, 4)

    def test_agent_action_works_with_different_state_signals(self):
        """Tests whether agent method works with variety of state signals."""
        for state in [None, 3, [1, 2], np.array([[1.0, 2.0], [3.0, 4.0]])]:
            with self.subTest(state=state):
                agent = DiscreteRandomAgent(5)
                action = agent.action(state)
                self.assertGreaterEqual(action, 0)

    def test_agent_reward(self):
        """Tests whether agent has reward method.

        Note that for a random agent, this doesn't do anything, but method
        should be provided for compatibility with other agents.
        """
        agent = DiscreteRandomAgent(5, random_state=32)
        agent.reward(3)
