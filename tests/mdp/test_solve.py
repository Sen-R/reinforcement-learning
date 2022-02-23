import pytest
from rl.mdp import FiniteMDP
from rl.mdp.solve import (
    exact_state_values,
    exact_optimum_state_values,
    iterative_policy_evaluation,
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
        initial_v = {s: 0.0 for s in gridworld.states}
        v = iterative_policy_evaluation(
            gridworld,
            gamma=0.9,
            pi=lambda a, s: 0.25,
            initial_v=initial_v,
            tol=1e-4,
        )
        assert round(v[(3, 1)], 1) == -0.4
        assert round(v[(4, 3)], 1) == -1.4
        assert round(v[(0, 1)], 1) == 8.8

    def test_maxiter_terminates_iteration(self, gridworld) -> None:
        with pytest.warns(UserWarning):
            iterative_policy_evaluation(
                gridworld,
                gamma=0.9,
                pi=lambda a, s: 0.25,
                initial_v={s: 0.0 for s in gridworld.states},
                tol=1e-10,
                maxiter=1,
            )
