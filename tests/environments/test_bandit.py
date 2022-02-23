import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from rl.environments.bandit import MultiArmedBandit, random_bandit


class TestBandit:
    def test_bandit_init(self) -> None:
        means = [1.0, 2.0, 3.0]
        sigmas = [1.0, 1.5, 0.5]
        b = MultiArmedBandit(means, sigmas)
        assert_array_equal(means, b.means)
        assert_array_equal(sigmas, b.sigmas)

    def test_bandit_k(self) -> None:
        b = MultiArmedBandit([1.0], [1.0])
        assert b.k == len(b.means)

    def test_bandit_act_functional(self) -> None:
        """Functional test of bandit.act.

        Note these tests are statistical, and may fail if the random seed
        is changed / RNG implementation is changed. Try a different seed
        or adjust the margin parameters if you are confident the implementation
        is correct.
        """
        # Set up a bandit with very different lever distributions
        means = [-100.0, 100.0]
        sigmas = [1.0, 2.0]
        bandit = MultiArmedBandit(means, sigmas, random_state=32)

        # Choose one of the levers, draw 100 samples and check if mean and
        # std dev are as expected
        chosen_lever = 1
        n_samples = 100
        rewards = np.array(
            [bandit.act(chosen_lever) for _ in range(n_samples)]
        )
        z_scores = (rewards - 100.0) / 2.0  # using mean and sigma for lever 1

        # Mean of z_scores should be normally distributed with zero e.v. and
        # 1/n_samples variance
        mean_margin = 0.5
        assert np.abs(np.mean(z_scores)) < mean_margin / np.sqrt(n_samples)

        # Sum of z_scores ** 2 should be chi2 distributed with dof n_samples.
        # For large dof this is approximately N(n_samples, 2*n_samples)
        chi2_margin = 1.0
        assert np.abs(
            np.sum(z_scores**2) - n_samples
        ) < chi2_margin * np.sqrt(2.0 * n_samples)

    def test_bandit_act(self) -> None:
        """Test assumes implementation uses np.random.default_rng. Kept here
        for now as a regression test, but is a candidate for removal (suggest
        replacing with a unit test mocking the RNG."""
        b = MultiArmedBandit([0.0, -0.5], [1.0, 2.0], random_state=42)
        actions = [1, 0, 1]
        rewards = [b.act(a) for a in actions]
        expected = [0.1094341595, -1.0399841062, 1.000902391]
        assert_almost_equal(rewards, expected)

    def test_optimal_action_returns_optimal_lever_idx(self) -> None:
        means = [0.0, 2.0, 1.0]
        sigmas = [10.0, 1.0, 0.5]  # unimportant
        optimal_lever = 1  # argmax(means)
        b = MultiArmedBandit(means, sigmas)
        assert b.optimal_action() == optimal_lever

    def test_state_updater_unset_by_default(self) -> None:
        b = MultiArmedBandit(means=[1.0], sigmas=[0.0])
        assert b.state_updater is None

    def test_state_updater_correctly_applied(self) -> None:
        # Configure bandit with simple state updating function
        means = [0.0, 0.0]
        sigmas = [1.0, 1.0]

        def state_updater(means, sigmas):
            return means + 1.0, sigmas * 0.5

        b = MultiArmedBandit(
            means=means, sigmas=sigmas, state_updater=state_updater
        )

        # Check update correctly happens
        b.act(0)
        assert_array_equal(b.means, [1.0, 1.0])
        assert_array_equal(b.sigmas, [0.5, 0.5])

        # Check call to reset correctly resets to initial state
        b.reset()
        assert_array_equal(b.means, [0.0, 0.0])
        assert_array_equal(b.sigmas, [1.0, 1.0])

    def test_rewards_constant_when_random_walk_params_unset(self) -> None:
        means = [0.0, 0.0]
        sigmas = [1.0, 1.0]
        b = MultiArmedBandit(means=means, sigmas=sigmas)
        b.act(0)
        assert_array_equal(b.means, means)
        assert_array_equal(b.sigmas, sigmas)

    def test_state_property(self) -> None:
        means = [0.0, 1.0]
        sigmas = [2.0, 3.0]
        b = MultiArmedBandit(means=means, sigmas=sigmas)
        state = b.state
        assert_array_equal(state["means"], means)
        assert_array_equal(state["sigmas"], sigmas)


class TestRandomBandit:
    def test_random_bandit(self) -> None:
        k = 3
        mean_loc, mean_scale = 5.0, 0.5
        sigma_loc, sigma_scale = 1.0, 2.0
        bandit = random_bandit(
            k,
            mean_params=(mean_loc, mean_scale),
            sigma_params=(sigma_loc, sigma_scale),
            random_state=42,
        )
        expected_means = np.array([5.15235854, 4.48000795, 5.3752256])
        expected_sigmas = np.array([2.88112943, -2.90207038, -1.60435901])
        assert_almost_equal(bandit.means, expected_means)
        assert_almost_equal(bandit.sigmas, expected_sigmas)
