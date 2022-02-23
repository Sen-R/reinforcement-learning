import warnings
import pytest
from rl.mdp import FiniteMDP
from rl.mdp.solve import (
    exact_state_values,
    exact_optimum_state_values,
    iterative_policy_evaluation,
    policy_iteration,
)


class TestExactStateValues:
    def test_state_values_for_sutton_barto_gridworld(
        self, gridworld: FiniteMDP
    ) -> None:
        """Compares outputs of `exact_state_values` against results given
        in the textbook where the specified gridworld is introduced."""
        v = exact_state_values(gridworld, gamma=0.9, pi=lambda a, s: 0.25)
        assert round(v[(3, 1)], 1) == -0.4
        assert round(v[(4, 3)], 1) == -1.4
        assert round(v[(0, 1)], 1) == 8.8


class TestExactOptimumStateValues:
    def test_optimum_state_values_for_sutton_barto_gridworld(
        self, gridworld: FiniteMDP
    ) -> None:
        """Compares outputs of `exact_optimum_state_values` against results
        given in the textbook where this gridworld is introduced."""
        v_star = exact_optimum_state_values(gridworld, gamma=0.9)
        assert round(v_star[(3, 1)], 1) == 17.8
        assert round(v_star[(4, 3)], 1) == 13.0
        assert round(v_star[(0, 1)], 1) == 24.4


class TestIterativePolicyEvaluation:
    def test_iterative_policy_evaluation(self, gridworld) -> None:
        v = {s: 0.0 for s in gridworld.states}
        niter = iterative_policy_evaluation(
            v,
            gridworld,
            gamma=0.9,
            pi=lambda a, s: 0.25,
            tol=1e-4,
        )
        assert niter > 0 and niter < 50  # should take less than 50 sweeps
        assert round(v[(3, 1)], 1) == -0.4
        assert round(v[(4, 3)], 1) == -1.4
        assert round(v[(0, 1)], 1) == 8.8

    def test_maxiter_terminates_iteration(self, gridworld) -> None:
        with pytest.warns(UserWarning):
            niter = iterative_policy_evaluation(
                {s: 0.0 for s in gridworld.states},
                gridworld,
                gamma=0.9,
                pi=lambda a, s: 0.25,
                tol=1e-10,
                maxiter=10,
            )
        assert niter == 10

    def test_can_set_tol_to_none_to_perform_fixed_number_of_iterations(
        self, gridworld
    ) -> None:
        # We catch warnings to ensure no warning is raised if tol is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            niter = iterative_policy_evaluation(
                {s: 0.0 for s in gridworld.states},
                gridworld,
                gamma=0.9,
                pi=lambda a, s: 0.25,
                tol=None,
                maxiter=5,
            )
        assert niter == 5


class TestPolicyIteration:
    def test_policy_iteration_functionality(self, gridworld) -> None:
        # Set up initial state value estimates and deterministic policy
        v = {s: 0.0 for s in gridworld.states}
        pi = {s: "n" for s in gridworld.states}

        # Perform policy iteration to refine both v and pi
        niter = policy_iteration(v, pi, gridworld, gamma=0.9, tol=1e-4)

        # Check whether policy matches optimal policy given in Sutton-Baro
        # for this gridworld
        assert niter > 0 and niter < 10  # should take less than 10 iterations
        assert pi[(0, 0)] == "e"
        assert pi[(1, 2)] in {"n", "w"}
        assert pi[(4, 1)] == "n"

    def test_maxiter_terminates_iteration(self, gridworld) -> None:
        with pytest.warns(UserWarning):
            niter = policy_iteration(
                {s: 0.0 for s in gridworld.states},
                {s: "n" for s in gridworld.states},
                gridworld,
                gamma=0.9,
                tol=1e-4,
                maxiter=2,
            )
        assert niter == 2
