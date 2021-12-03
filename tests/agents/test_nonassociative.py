import unittest
from numpy.testing import assert_array_equal, assert_array_almost_equal
from rl.agents.nonassociative import RewardAveragingEpsilonGreedyAgent
from rl.learningrate import SampleAverageLearningRate


class TestRewardAveragingEpsilonGreedyAgent(unittest.TestCase):
    def test_attributes(self) -> None:
        """Tests whether initial instance attributes are set correctly upon
        object initialisation."""
        # Parameters
        k = 3  # arbitrary
        agent = RewardAveragingEpsilonGreedyAgent(k)

        # Test
        self.assertEqual(agent.n_actions, k)
        assert_array_equal(agent.Q, [0.0] * k)  # Q initialised to zero
        assert_array_equal(agent.action_counts, [0] * k)  # counts are zero
        self.assertEqual(agent.epsilon, 0.0)  # default epsilon is zero
        self.assertIsInstance(agent.alpha, SampleAverageLearningRate)

    def test_setting_initial_action_values(self) -> None:
        """Tests whether can correctly set initial action values to something
        other than the default value."""
        # Parameters
        k = 3  # arbitrary
        init_q = [5.0, 2.0, 3.0]  # set to something other than zero

        # Test
        agent = RewardAveragingEpsilonGreedyAgent(
            k, initial_action_values=init_q
        )
        assert_array_equal(agent.Q, init_q)

    def test_setting_epsilon(self) -> None:
        # Parameters
        k, eps = 3, 0.5

        # Test
        agent = RewardAveragingEpsilonGreedyAgent(k, epsilon=eps)
        self.assertEqual(agent.epsilon, eps)

    def test_returns_epsilon_greedy_action(self) -> None:
        """Tests whether calls to `_get_action_selector` yields the correct
        epsilon greedy action selector."""
        # Parameters
        Q = [0.0, 1.0, 0.0]  # Chosen so that action 1 is best action
        epsilon = 0.2  # arbitrary
        a_opt = 1  # i.e. argmax(Q)
        k = 3  # i.e. len(Q)

        # Test:
        # Create an agent, setting initial action values to Q defined above,
        # then test whether _get_action_selector yields a correctly configured
        # epsilon-greedy action selector
        agent = RewardAveragingEpsilonGreedyAgent(
            k, epsilon=epsilon, initial_action_values=Q
        )
        action_selector = agent._get_action_selector()
        self.assertEqual(action_selector.epsilon, epsilon)
        self.assertEqual(action_selector.preferred.chosen_action, a_opt)
        self.assertEqual(action_selector.noise.n_actions, k)

    def test_reward_updates_observation_counts(self) -> None:
        """Tests whether calling reward results in observation counts
        being correctly updated."""
        # Parameters
        Q = [0.0, 1.0, 0.0]  # Chosen so that action 1 is best
        k = 3  # len(Q)
        a_opt = 1  # argmax(Q)
        expected_counts_after_reward = [0, 1, 0]  # only action 1 taken
        epsilon = 0  # To ensure greedy action selection
        reward = 0  # arbitrary

        # Test:
        # Construct agent, setting initial action values to Q defined above
        # and let it take a single action (should be action 1). Then send the
        # agent a reward before checking agent.action_counts: these counts
        # should correspond to the expected counts defined above.
        agent = RewardAveragingEpsilonGreedyAgent(
            k, epsilon=epsilon, initial_action_values=Q
        )
        action = agent.action(None)
        self.assertEqual(action, a_opt)  # Sanity check, already tested
        agent.reward(reward)
        assert_array_equal(agent.action_counts, expected_counts_after_reward)

    def test_reward_updates_action_values(self) -> None:
        """Tests whether calling reward results in action values being
        being currectly updated (using soft updates)."""
        # Parameters
        Q = [0.0, 1.0, 0.0]  # Chosen so that action 1 is best
        k = 3  # len(Q)
        a_opt = 1  # argmax(Q)

        def lr(action_count):
            # arbitrary schedule, but should be function of action_count
            return 0.1 / (action_count + 1)

        epsilon = 0  # Ensure greedy action selection
        rewards = [0.5, 1.5]
        expected_Qs_after_updates = [[0.0, 0.95, 0.0], [0.0, 0.9775, 0.0]]

        # Test:
        # Construct the agent, setting initial action values to Q defined
        # above. Let the agent take two action-reward cycles, feeding in
        # rewards defined above and check that Q values update as expected
        # (expected results defined above).
        agent = RewardAveragingEpsilonGreedyAgent(
            k,
            epsilon=epsilon,
            initial_action_values=Q,
            learning_rate_schedule=lr,
        )
        for reward, Q_after_update in zip(rewards, expected_Qs_after_updates):
            a = agent.action(None)
            self.assertEqual(a, a_opt)  # sanity check, already tested
            agent.reward(reward)
            assert_array_almost_equal(agent.Q, Q_after_update)
