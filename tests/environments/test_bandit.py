import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from rl.environments.bandit import MultiArmBandit, random_bandit


class TestBandit(unittest.TestCase):
    def test_bandit_init(self):
        means = [1.0, 2.0, 3.0]
        sigmas = [1.0, 1.5, 0.5]
        b = MultiArmBandit(means, sigmas)
        self.assertEqual(means, b.means)
        self.assertEqual(sigmas, b.sigmas)

    def test_bandit_k(self):
        b = MultiArmBandit([1.0], [1.0])
        self.assertEqual(b.k, len(b.means))

    def test_bandit_act(self):
        b = MultiArmBandit([0.0, -0.5], [1.0, 2.0], random_state=42)
        actions = [1, 0, 1]
        rewards = [b.act(a) for a in actions]
        expected = [0.1094341595, -1.0399841062, 1.000902391]
        assert_almost_equal(rewards, expected)


class TestRandomBandit(unittest.TestCase):
    def test_random_bandit(self):
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
