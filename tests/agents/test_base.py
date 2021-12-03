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

    def test_last_action_corresponds_to_chosen_action(self):
        """Tests whether agent.last_action matches the action last provided
        by the agent."""
        # Parameters
        actions = [0, 1, 2]  # arbitrary actions to call
        action_selectors = [DeterministicActionSelector(a) for a in actions]

        # Test:
        # Create mock agent that for which _get_action_selector
        # returns a sequence of deterministic action selections and
        # check whether calls to action correctly update the agent's
        # last_action field.
        MockAgent = create_autospec(Agent)
        agent = MockAgent()
        agent._get_action_selector.side_effect = action_selectors
        for a in actions:
            chosen_action = Agent.action(agent, state=None)
            self.assertEqual(chosen_action, a)  # sanity check
            self.assertEqual(agent.last_action, chosen_action)
