from typing import Callable, Dict, Optional
import numpy as np
from scipy import optimize  # type: ignore
from ._types import State, Action
from .base import FiniteMDP


def exact_state_values(
    mdp: FiniteMDP[Action, State],
    gamma: float,
    pi: Callable[[Action, State], float],
) -> Dict[State, float]:
    """Returns state values for given policy.

    This function directly solves the (linear) Bellman equation to calculate
    the state value function for the policy represented by `pi`.

    Args:
      mdp: MDP for which state values are being calculated
      gamma: discount factor
      pi: conditional probabilities for actions given states, encoding the
        policy to be evaluated

    Returns:
      `dict` mapping states to state values
    """
    A, b = mdp.backup_policy_values_operator(gamma, pi)
    v = np.linalg.solve(np.eye(len(mdp.states)) - A, b)
    return {mdp.i2s(idx): value for idx, value in enumerate(v)}


def exact_optimum_state_values(
    mdp: FiniteMDP[Action, State], gamma: float, tol: Optional[float] = None
) -> Dict[State, float]:
    """Returns state values for an optimal policy for the given MDP.

    This function uses a non-linear solver to directly solve the Bellman
    optimality equation, to calculate state values for an optimal policy.

    Args:
      mdp: MDP for which optimal state values are being calculated
      gamma: discount factor

    Returns:
      `dict` mapping states to state values under an optimal policy
    """
    initial_guess = np.zeros(len(mdp.states))
    opt_result = optimize.root(
        lambda v_star: v_star - mdp.backup_optimal_values(v_star, gamma),
        x0=initial_guess,
        tol=tol,
    )

    if not opt_result.success:
        raise optimize.OptimizeWarning(
            "Root finding failed to find a solution", opt_result
        )

    return {mdp.i2s(idx): val for idx, val in enumerate(opt_result.x)}
