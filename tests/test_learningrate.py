import unittest
from rl.learningrate import (
    ConstantLearningRate,
    SampleAverageLearningRate,
)


class TestConstantLearningRate(unittest.TestCase):
    def test_function_is_correct(self):
        alpha = 0.01
        lr = ConstantLearningRate(alpha)
        for n in range(10):
            with self.subTest(n=n):
                self.assertEqual(lr(n), alpha)


class TestSampleAverageLearningRate(unittest.TestCase):
    def test_function_is_correct(self):
        lr = SampleAverageLearningRate()
        for n in range(10):
            with self.subTest(n=n):
                self.assertEqual(lr(n), 1. / (n + 1))
