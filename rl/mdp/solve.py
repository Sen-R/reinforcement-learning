from typing import Callable, Dict, Optional, MutableMapping
from warnings import warn
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


def iterative_policy_evaluation(
    v: MutableMapping[State, float],
    mdp: FiniteMDP[Action, State],
    gamma: float,
    pi: Callable[[Action, State], float],
    tol: Optional[float],
    maxiter: int = 100,
) -> int:
    """Applies iterative policy evaluation to refine provided state value
    estimates.

    Args:
      v: initial estimates of state values which are refined in place
      mdp: MDP for which state values are being estimated
      gamma: discount factor
      pi: conditional probabilities for actions given states, encoding the
        policy being evaluated
      tol: iteration terminates when maximum absolute change in state value
        function falls below this value, alternative set to `None` and
        iteration will proceed until maxiter is reached
      maxiter: iteration terminates when the number of sweeps through the
        MDP's state space reaches this value

    Returns:
      niter: number of sweeps of the state space that were completed
    """
    for niter in range(1, maxiter + 1):
        delta_v = 0.0  # tracks biggest change to v so far
        for s in mdp.states:
            v_old = v[s]
            v[s] = mdp.backup_single_state_value(s, v, gamma, pi)
            delta_v = max(delta_v, abs(v[s] - v_old))
        if tol is not None and delta_v < tol:
            break
    else:
        # Loop completed normally implying maxiter was reached. If tol is
        # not None, these means the solution has not converged to the tolerance
        # expected.

        if tol is not None:
            warn(
                "`maxiter` sweeps were completed before solution converged to "
                "within desired tolerance, try increasing either `maxiter` or "
                "`tol`"
            )
    return niter


def policy_iteration(
    v: MutableMapping[State, float],
    pi: MutableMapping[State, Action],
    mdp: FiniteMDP[Action, State],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs policy iteration to refine policy `pi` and corresponding
    state value estimates `v`.

    Args:
      v: mapping from states to state values for current policy
      pi: mapping from states to actions, encoding a deterministic policy
      mdp: MDP for which optimal policy is being searched
      gamma: discount factor
      tol: tolerance for embedded policy evaluation component of each policy
        iteration to converge
      maxiter: maximum number of policy iterations to perform

    Returns:
      niter: number of sweeps of the state space that were performed
    """
    for niter in range(1, maxiter + 1):
        policy_stable = True
        for s in mdp.states:
            old_action = pi[s]
            pi[s], _ = mdp.backup_single_state_optimal_action(s, v, gamma)
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break
        iterative_policy_evaluation(
            v, mdp, gamma, lambda a, s: (a == pi[s]), tol
        )
    else:
        # maxiter reached but policy not yet stable, so warn
        warn("maxiter reached but policy not yet stable")
    return niter


def value_iteration(
    v: MutableMapping[State, float],
    mdp: FiniteMDP[Action, State],
    gamma: float,
    tol: float,
    maxiter: int = 100,
) -> int:
    """Performs value iteration to evolve the supplied state value mapping `v`
    into state values for an optimal policy.

    Args:
      v: initial state value estimates that will be updated in place
      mdp: MDP for which optimal state values are to be estimated
      gamma: discount factor
      tol: if the maximum absolute change of state values in one sweep is
        below this value, then iteration will stop
      maxiter: if the number of sweeps of value iteration reaches this value
        then iteration will stop

    Returns:
      niter: number of sweeps of the state space that took place
    """
    for niter in range(1, maxiter + 1):
        delta_v = 0.0
        for s in mdp.states:
            v_old = v[s]
            _, v[s] = mdp.backup_single_state_optimal_action(s, v, gamma)
            delta_v = max(delta_v, abs(v_old - v[s]))
        if delta_v < tol:
            break
    else:
        warn("`maxiter` reached without convergence within tolerance `tol`")
    return niter
