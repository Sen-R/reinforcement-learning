"""Types used in the rl package."""

from typing import Callable


LearningRateSchedule = Callable[[int], float]
