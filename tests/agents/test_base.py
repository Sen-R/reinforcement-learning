import unittest
from unittest.mock import create_autospec
from rl.agents.base import Agent
from rl.agents.action_selector import DeterministicActionSelector


class TestBase(unittest.TestCase):
    def test_action_method(self):
        """Tests whether `action` method correctly calls `_get_action_selector`
        to determine which action to return"""
        MockAgent = create_autospec(Agent)
        agent = MockAgent()
        state = 0
        expected_action = 64
        agent._get_action_selector.return_value = DeterministicActionSelector(
            expected_action
        )
        chosen_action = Agent.action(agent, state=state)
        self.assertEqual(chosen_action, expected_action)
        agent._get_action_selector.assert_called_with(state)
