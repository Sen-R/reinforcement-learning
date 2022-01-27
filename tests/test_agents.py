import numpy as np
from rl.policies.value_learning import RewardAveragingPolicy
from rl.policies.action_selection_strategy import EpsilonGreedy, UCB
from rl.agents import (
    EpsilonGreedyRewardAveragingAgent,
    UCBRewardAveragingAgent,
)
from rl.learningrate import SampleAverageLearningRate


class TestEpsilonGreedyRewardAveragingAgent:
    def test_init(self) -> None:
        # Create agent
        n_actions = 3
        epsilon = 0.01
        rng = np.random.default_rng(42)

        def learning_rate_schedule(_):
            return 0.1

        initial_action_values = [5.0] * n_actions
        agent = EpsilonGreedyRewardAveragingAgent(
            epsilon,
            n_actions,
            learning_rate_schedule=learning_rate_schedule,
            initial_action_values=initial_action_values,
            random_state=42,
        )

        # Test whether agent is set up correctly
        policy = agent.policy
        assert isinstance(policy, RewardAveragingPolicy)
        assert isinstance(policy.action_selection_strategy, EpsilonGreedy)
        assert policy.action_selection_strategy.epsilon == epsilon
        assert len(policy.Q) == n_actions
        assert policy.alpha == learning_rate_schedule
        assert policy.Q == initial_action_values
        assert (
            policy.action_selection_strategy._rng.bit_generator.state
            == rng.bit_generator.state
        )

    def test_defaults(self) -> None:
        # Create agent
        epsilon = 0.1
        n_actions = 2
        agent = EpsilonGreedyRewardAveragingAgent(epsilon, n_actions)
        policy = agent.policy
        assert isinstance(policy, RewardAveragingPolicy)
        assert isinstance(policy.alpha, SampleAverageLearningRate)
        assert policy.Q == [0.0] * 2


class TestUCBRewardAveragingAgent:
    def test_init(self) -> None:
        # Create agent
        c = 2
        n_actions = 2

        def learning_rate_schedule(_):
            return 0.1

        initial_action_values = [1.0, 2.0]

        agent = UCBRewardAveragingAgent(
            c,
            n_actions,
            learning_rate_schedule=learning_rate_schedule,
            initial_action_values=initial_action_values,
        )

        # Check created agent is of right type and contains right fields
        policy = agent.policy
        assert isinstance(policy, RewardAveragingPolicy)
        assert isinstance(policy.action_selection_strategy, UCB)
        assert policy.action_selection_strategy.c == c
        assert len(policy.Q) == n_actions
        assert policy.alpha == learning_rate_schedule
        assert policy.Q == initial_action_values

    def test_defaults(self) -> None:
        # Create agent
        c = 2
        n_actions = 2
        agent = UCBRewardAveragingAgent(c, n_actions)
        policy = agent.policy
        assert isinstance(policy, RewardAveragingPolicy)
        assert isinstance(policy.alpha, SampleAverageLearningRate)
        assert policy.Q == [0.0] * 2

    def test_optional_random_state_argument_is_allowed(self) -> None:
        # For compatibility with other agents that do rely on randomness
        c = 2
        n_actions = 2
        random_state = 42
        UCBRewardAveragingAgent(c, n_actions, random_state=random_state)
