from typing import Iterable
import pytest
from numpy.testing import assert_almost_equal
from rl.mdp._types import NextStateProbabilityTable
from rl.mdp import FiniteMDP
from rl.mdp.gamblersproblem import GamblersProblem


@pytest.fixture
def gp() -> FiniteMDP:
    return GamblersProblem(goal=100, p_h=0.25)


class TestGamblersProblem:
    def test_states_property(self, gp: FiniteMDP) -> None:
        expected = list(range(0, 101))
        actual = list(gp.states)
        assert actual == expected

    @pytest.mark.parametrize(
        "state,expected_actions",
        [
            (3, range(0, 4)),
            (90, range(0, 11)),
            (0, [0]),
            (100, [0]),
        ],
    )
    def test_actions_method(
        self, gp: FiniteMDP, state: int, expected_actions: Iterable[int]
    ) -> None:
        actual_actions = gp.actions(state)
        assert actual_actions, list(expected_actions)

    @pytest.mark.parametrize("state,index", [(0, 0), (13, 13), (100, 100)])
    def test_s2i_i2s(self, gp: FiniteMDP, state: int, index: int) -> None:
        assert gp.s2i(state) == index
        assert gp.i2s(index) == state

    @pytest.mark.parametrize(
        "state,action,des_ns_ptable,des_exp_r",
        [
            (25, 10, ((15, 35), (0.75, 0.25)), 0.0),
            (0, 0, ((0,), (1.0,)), 0.0),
            (13, 0, ((13,), (1.0,)), 0.0),
            (75, 25, ((50, 100), (0.75, 0.25)), 0.25),
            (100, 0, ((100,), (1.0,)), 0.0),
        ],
    )
    def test_next_states_and_rewards(
        self,
        gp: FiniteMDP,
        state: int,
        action: int,
        des_ns_ptable: NextStateProbabilityTable[int],
        des_exp_r: float,
    ) -> None:
        act_ns_ptable, act_exp_r = gp.next_states_and_rewards(state, action)
        assert_almost_equal(act_exp_r, des_exp_r)
        assert act_ns_ptable[0] == des_ns_ptable[0]
        assert_almost_equal(act_ns_ptable[1], des_ns_ptable[1])
