import unittest
from unittest.mock import create_autospec
from rl.agent import Agent
from rl.action_selector import DeterministicActionSelector


def mock_agent():
    MockAgentType = create_autospec(Agent)
    return MockAgentType()


class TestBase(unittest.TestCase):
    def test_action_method(self) -> None:
        """Tests whether `action` method correctly calls `_get_action_selector`
        to determine which action to return"""
        agent = mock_agent()
        state = 0
        expected_action = 64
        agent._get_action_selector.return_value = DeterministicActionSelector(
            expected_action
        )
        chosen_action = Agent.action(agent, state=state)
        self.assertEqual(chosen_action, expected_action)
        agent._get_action_selector.assert_called_with(state)

    def test_last_action_corresponds_to_chosen_action(self) -> None:
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
        agent = mock_agent()
        agent._get_action_selector.side_effect = action_selectors
        for a in actions:
            chosen_action = Agent.action(agent, state=None)
            self.assertEqual(chosen_action, a)  # sanity check
            self.assertEqual(agent.last_action, chosen_action)
            Agent.reward(agent, 0)  # arbitrary reward to advance loop

    def test_last_state_corresponds_to_last_received_state(self) -> None:
        """Tests whether agent.last_state corresponds to the last state
        received when action method was called."""
        # Parameters
        states = [1, 2]  # arbitrary, but changing

        # Test:
        # Create mock agent, sending it a sequence of states and checking
        # that last_state corresponds correctly to the last sent state after
        # action is called
        agent = mock_agent()
        for s in states:
            Agent.action(agent, s)
            self.assertEqual(agent.last_state, s)
            Agent.reward(agent, 0)  # arbitrary reward

    def test_reward_resets_last_state_and_last_action(self) -> None:
        """Tests whether calling the reward method resets the last_state
        and last_action fields in agent."""
        state = 1  # arbitrary
        agent = mock_agent()

        # Call Agent.action and first sanity check that last_* fields are set.
        Agent.action(agent, state=state)
        self.assertTrue(hasattr(agent, "last_action"))
        self.assertTrue(hasattr(agent, "last_state"))

        # Then call Agent.reward and check that these fields no longer set.
        Agent.reward(agent, reward=0)
        self.assertFalse(hasattr(agent, "last_action"))
        self.assertFalse(hasattr(agent, "last_state"))

    def test_reward_cannot_be_called_before_action_ever_called(self) -> None:
        agent = mock_agent()
        with self.assertRaises(RuntimeError):
            Agent.reward(agent, reward=0)

    def test_reward_cannot_be_called_twice_in_a_row(self) -> None:
        agent = mock_agent()
        Agent.action(agent, state=None)
        Agent.reward(agent, reward=0.0)
        with self.assertRaises(RuntimeError):
            Agent.reward(agent, reward=0.0)  # raises as action not called
        Agent.action(agent, state=None)
        Agent.reward(agent, reward=0.0)  # OK now that action has been called

    def test_action_cannot_be_called_twice_in_a_row(self) -> None:
        agent = mock_agent()
        Agent.action(agent, state=None)
        with self.assertRaises(RuntimeError):
            Agent.action(agent, state=None)  # raises as reward not called
        Agent.reward(agent, reward=0.0)
        Agent.action(agent, state=None)  # OK now that reward has been called
