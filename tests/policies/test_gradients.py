from numpy.random import default_rng
from numpy.testing import assert_array_almost_equal, assert_array_equal
from rl.policies.gradients import GradientBandit
from rl.action_selectors import DiscreteActionSelector


class TestGradientBandit:
    def test_call(self) -> None:
        alpha = 0.1
        n_actions = 2
        H = [-0.1, 0.2]
        p = [0.42555748, 0.57444252]  # soft-max of H
        policy = GradientBandit(alpha, n_actions, initial_preferences=H)
        action_selector = policy(state=0)
        assert isinstance(action_selector, DiscreteActionSelector)
        assert_array_almost_equal(action_selector.p, p)

    def test_update(self) -> None:
        # Setup policy
        alpha = 0.2
        n_actions = 2
        H = [-0.1, 0.2]
        p = [0.42555748, 0.57444252]
        policy = GradientBandit(alpha, n_actions, initial_preferences=H)

        # Run single update step
        policy.update(state=0, action=1, reward=3.0)

        # After first update step, H vector should be unchanged (using
        # default baseline of average rewards so far)
        assert_array_equal(policy.H, H)

        # Perform second update step
        policy.update(state=0, action=0, reward=2.0)

        # Manually calculate what we expect new H vector to be
        assert policy.baseline == 2.5  # mean of previous rewards
        new_H = [
            -0.1 + 0.2 * (2.0 - 2.5) * (1.0 - p[0]),
            0.2 - 0.2 * (2.0 - 2.5) * p[1],
        ]
        assert_array_almost_equal(policy.H, new_H)

    def test_state_returns_expected_dict(self) -> None:
        policy = GradientBandit(0.1, 2, initial_preferences=[1.0, 2.0])
        state = policy.state
        assert list(state.keys()) == ["H"]
        assert_array_equal(state["H"], policy.H)
        assert state["H"] is not policy.H  # ensure it's a copy

    def test_set_random_state(self) -> None:
        seed = 42
        policy = GradientBandit(0.1, 3, random_state=42)
        action_selector = policy(state=0)
        assert action_selector._rng.random() == default_rng(seed).random()

    def test_default_preferences_are_zero(self) -> None:
        policy = GradientBandit(0.1, 3)
        assert_array_equal(policy.H, [0.0] * 3)

    def test_number_as_baseline_sets_it_to_a_constant(self) -> None:
        # Set up a policy with arbitrary parameters and baseline arg
        # set to a float
        init_H = [-0.1, 0.2]
        init_p = [0.42555748, 0.57444252]  # soft-max of H
        policy = GradientBandit(
            0.1, 2, initial_preferences=init_H, baseline=-3.0
        )

        # Update the policy and see if H evolves as expected
        policy.update(state=0, action=1, reward=4.0)
        assert policy.baseline == -3.0
        new_H = [
            -0.1 - 0.1 * (4.0 + 3.0) * init_p[0],
            0.2 + 0.1 * (4.0 + 3.0) * (1 - init_p[1]),
        ]
        assert_array_almost_equal(new_H, policy.H)

    def test_default_baseline_is_sample_averaged_rewards(self) -> None:
        policy = GradientBandit(0.1, 2)
        assert policy.baseline == 0
        policy.update(state=0, action=1, reward=1.0)
        assert policy.baseline == 1.0
        policy.update(state=0, action=1, reward=2.0)
        assert policy.baseline == 1.5
