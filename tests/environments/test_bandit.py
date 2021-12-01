import unittest
from numpy.testing import assert_almost_equal
from rlbook.environments.bandit import MultiArmBandit


class TestBandit(unittest.TestCase):
    def test_bandit_setup(self):
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
