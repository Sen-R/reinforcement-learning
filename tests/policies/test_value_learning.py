from unittest.mock import Mock
from numpy.testing import assert_array_equal, assert_array_almost_equal
from rl.policies.value_learning import RewardAveragingPolicy
from rl.policies.action_selection_strategy import EpsilonGreedy
from rl.learningrate import SampleAverageLearningRate


class TestRewardAveragingPolicy:
    def test_attributes(self) -> None:
        """Tests whether initial instance attributes are set correctly upon
        object initialisation."""
        # Parameters
        k = 3  # arbitrary
        policy = RewardAveragingPolicy(
            n_actions=k,
            action_selection_strategy=EpsilonGreedy(),
        )

        # Test
        assert_array_equal(policy.Q, [0.0] * k)  # Q initialised to zero
        assert_array_equal(policy.action_counts, [0] * k)  # counts are zero
        assert isinstance(policy.alpha, SampleAverageLearningRate)

    def test_action_selection_strategy_called(self) -> None:
        """Tests whether the policy's __call__ method correctly calls in
        turn the innder action selection strategy to return an action
        selector."""
        k = 3  # arbitrary
        init_Q = [1.0, 2.0, 3.0]  # arbitrary
        mock_strategy = Mock()
        policy = RewardAveragingPolicy(
            n_actions=k,
            action_selection_strategy=mock_strategy,
            initial_action_values=init_Q,
        )
        fake_state = 0  # arbitrary
        policy(fake_state)  # should have called strategy
        mock_strategy.assert_called_with(policy.Q, policy.action_counts)

    def test_setting_initial_action_values(self) -> None:
        """Tests whether can correctly set initial action values to something
        other than the default value."""
        # Parameters
        k = 3  # arbitrary
        init_q = [5.0, 2.0, 3.0]  # set to something other than zero

        # Test
        policy = RewardAveragingPolicy(
            k,
            action_selection_strategy=EpsilonGreedy(),
            initial_action_values=init_q,
        )
        assert_array_equal(policy.Q, init_q)

    def test_reward_updates_observation_counts(self) -> None:
        """Tests whether calling reward results in observation counts
        being correctly updated."""
        # Parameters
        k = 3  # arbitrary
        state = 0  # arbitrary
        action = 1  # arbitrary
        reward = 0.0  # arbitrary
        expected_counts_after_reward = [0, 1, 0]  # only action 1 taken

        # Test:
        # Construct policy, and send state-action-reward update. Check
        # action counts are as expected (given that a single experience for
        # action 1 is all that has been provided).
        policy = RewardAveragingPolicy(
            n_actions=k, action_selection_strategy=EpsilonGreedy()
        )
        policy.update(state, action, reward)
        assert_array_equal(policy.action_counts, expected_counts_after_reward)

    def test_reward_updates_action_values(self) -> None:
        """Tests whether calling reward results in action values being
        being currectly updated (using soft updates)."""
        # Parameters
        k = 3  # len(Q)
        state = 0  # arbitrary
        action = 1  # arbitrary
        rewards = [0.5, 1.5]  # arbitrary but changing
        Q = [0.0, 1.0, 0.0]

        def lr(action_count):
            # arbitrary schedule, but should be function of action_count
            return 0.1 / (action_count + 1)

        expected_Qs_after_updates = [[0.0, 0.95, 0.0], [0.0, 0.9775, 0.0]]

        # Test:
        # Construct the policy, setting initial action values to Q defined
        # above. Let the policy take two state-action-reward updates, feeding
        # rewards defined above and check that Q values update as expected
        # (expected results defined above).
        policy = RewardAveragingPolicy(
            k,
            action_selection_strategy=EpsilonGreedy(),
            initial_action_values=Q,
            learning_rate_schedule=lr,
        )
        for reward, Q_after_update in zip(rewards, expected_Qs_after_updates):
            policy.update(state, action, reward)
            assert_array_almost_equal(policy.Q, Q_after_update)

    def test_state_attr_returns_action_values_and_counts(self) -> None:
        policy = RewardAveragingPolicy(
            3, action_selection_strategy=EpsilonGreedy()
        )
        policy_state = policy.state

        # Check state dictionary has expected values
        assert policy_state == {
            "Q": policy.Q,
            "action_counts": policy.action_counts,
        }

        # However returned state should contain copies
        assert policy_state["Q"] is not policy.Q
        assert policy_state["action_counts"] is not policy.action_counts
