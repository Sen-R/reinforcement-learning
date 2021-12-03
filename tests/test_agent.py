import unittest
from unittest.mock import create_autospec
from .fakes import fake_agent, FakePolicy
from rl.agent import Agent
from rl.action_selector import DeterministicActionSelector


class TestBase(unittest.TestCase):
    def test_action_method(self) -> None:
        """Tests whether `action` method correctly calls the policy
        to determine which action to return and returns that action"""
        state = 0  # arbitrary
        expected_action = 64  # arbitrary
        policy = create_autospec(FakePolicy(0))
        policy.return_value = DeterministicActionSelector(expected_action)
        agent = Agent(policy)
        chosen_action = agent.action(state)
        self.assertEqual(chosen_action, expected_action)
        policy.assert_called_with(state)

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
        policy = create_autospec(FakePolicy(0))
        policy.side_effect = action_selectors
        agent = Agent(policy)
        for a in actions:
            chosen_action = agent.action(state=None)
            self.assertEqual(chosen_action, a)  # sanity check
            self.assertEqual(agent.last_action, chosen_action)
            agent.reward(0)  # arbitrary reward to advance loop

    def test_last_state_corresponds_to_last_received_state(self) -> None:
        """Tests whether agent.last_state corresponds to the last state
        received when action method was called."""
        # Parameters
        states = [1, 2]  # arbitrary, but changing

        # Test:
        # Create mock agent, sending it a sequence of states and checking
        # that last_state corresponds correctly to the last sent state after
        # action is called
        agent = fake_agent()
        for s in states:
            agent.action(s)
            self.assertEqual(agent.last_state, s)
            agent.reward(0)  # arbitrary reward

    def test_reward_resets_last_state_and_last_action(self) -> None:
        """Tests whether calling the reward method resets the last_state
        and last_action fields in agent."""
        state = 1  # arbitrary
        agent = fake_agent()

        # Call Agent.action and first sanity check that last_* fields are set.
        agent.action(state=state)
        self.assertTrue(hasattr(agent, "last_action"))
        self.assertTrue(hasattr(agent, "last_state"))

        # Then call Agent.reward and check that these fields no longer set.
        agent.reward(reward=0)
        self.assertFalse(hasattr(agent, "last_action"))
        self.assertFalse(hasattr(agent, "last_state"))

    def test_reward_cannot_be_called_before_action_ever_called(self) -> None:
        agent = fake_agent()
        with self.assertRaises(RuntimeError):
            agent.reward(reward=0)

    def test_reward_cannot_be_called_twice_in_a_row(self) -> None:
        agent = fake_agent()
        agent.action(state=None)
        agent.reward(reward=0.0)
        with self.assertRaises(RuntimeError):
            agent.reward(reward=0.0)  # raises as action not called
        agent.action(state=None)
        agent.reward(reward=0.0)  # OK now that action has been called

    def test_action_cannot_be_called_twice_in_a_row(self) -> None:
        agent = fake_agent()
        agent.action(state=None)
        with self.assertRaises(RuntimeError):
            agent.action(state=None)  # raises as reward not called
        agent.reward(reward=0.0)
        agent.action(state=None)  # OK now that reward has been called
