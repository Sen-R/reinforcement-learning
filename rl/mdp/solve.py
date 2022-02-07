from typing import Callable, Dict
import numpy as np
from ._types import State, Action
from .base import FiniteMDP


def exact_state_values(
    mdp: FiniteMDP[Action, State],
    gamma: float,
    pi: Callable[[Action, State], float],
) -> Dict[State, float]:
    A, b = mdp.bellman_operator(gamma, pi)
    v = np.linalg.solve(np.eye(len(mdp.states)) - A, b)
    return {mdp.i2s(idx): value for idx, value in enumerate(v)}
