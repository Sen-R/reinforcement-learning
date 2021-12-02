import unittest
from rl.utils import soft_update


class TestSoftUpdate(unittest.TestCase):
    def test_function_is_correct(self):
        alpha = 0.2
        current = 1.0
        target = 1.5
        expected = 1.1  # (1 - alpha) * current + alpha * target
        self.assertEqual(soft_update(current, target, alpha), expected)
