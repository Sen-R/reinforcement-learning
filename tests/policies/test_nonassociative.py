import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from rl.policies.nonassociative import RewardAveragingEpsilonGreedyPolicy
from rl.learningrate import SampleAverageLearningRate
from rl.action_selector import EpsilonGreedyActionSelector


class TestRewardAveragingEpsilonGreedyPolicy(unittest.TestCase):
    def test_attributes(self) -> None:
        """Tests whether initial instance attributes are set correctly upon
        object initialisation."""
        # Parameters
        k = 3  # arbitrary
        policy = RewardAveragingEpsilonGreedyPolicy(k)

        # Test
        assert_array_equal(policy.Q, [0.0] * k)  # Q initialised to zero
        assert_array_equal(policy.action_counts, [0] * k)  # counts are zero
        self.assertEqual(policy.epsilon, 0.0)  # default epsilon is zero
        self.assertIsInstance(policy.alpha, SampleAverageLearningRate)

    def test_setting_initial_action_values(self) -> None:
        """Tests whether can correctly set initial action values to something
        other than the default value."""
        # Parameters
        k = 3  # arbitrary
        init_q = [5.0, 2.0, 3.0]  # set to something other than zero

        # Test
        policy = RewardAveragingEpsilonGreedyPolicy(
            k, initial_action_values=init_q
        )
        assert_array_equal(policy.Q, init_q)

    def test_setting_epsilon(self) -> None:
        # Parameters
        k, eps = 3, 0.5

        # Test
        policy = RewardAveragingEpsilonGreedyPolicy(k, epsilon=eps)
        self.assertEqual(policy.epsilon, eps)

    def test_returns_epsilon_greedy_action(self) -> None:
        """Tests whether calls to `_get_action_selector` yields the correct
        epsilon greedy action selector."""
        # Parameters
        Q = [0.0, 1.0, 0.0]  # Chosen so that action 1 is best action
        epsilon = 0.2  # arbitrary
        a_opt = 1  # i.e. argmax(Q)
        k = 3  # i.e. len(Q)

        # Test:
        # Create an policy, setting initial action values to Q defined above,
        # then test whether _get_action_selector yields a correctly configured
        # epsilon-greedy action selector
        policy = RewardAveragingEpsilonGreedyPolicy(
            k, epsilon=epsilon, initial_action_values=Q
        )
        action_selector = policy()
        assert isinstance(action_selector, EpsilonGreedyActionSelector)
        self.assertEqual(action_selector.epsilon, epsilon)
        self.assertEqual(action_selector.preferred.chosen_action, a_opt)
        self.assertEqual(action_selector.noise.n_actions, k)

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
        policy = RewardAveragingEpsilonGreedyPolicy(k)
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
        policy = RewardAveragingEpsilonGreedyPolicy(
            k, initial_action_values=Q, learning_rate_schedule=lr
        )
        for reward, Q_after_update in zip(rewards, expected_Qs_after_updates):
            policy.update(state, action, reward)
            assert_array_almost_equal(policy.Q, Q_after_update)
