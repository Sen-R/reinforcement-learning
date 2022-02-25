from typing import List, Dict, Set, Tuple, Collection
import pytest
from numpy.testing import assert_almost_equal
from rl.mdp._types import TransitionsMapping, Policy
from rl.mdp.base import FiniteMDP


TState = str
TAction = str


class SimpleMDP(FiniteMDP[TState, TAction]):
    _states = ["A", "B", "C"]
    _actions = ["R", "L"]
    _rewards = {0.0, 1.0, -1.0}
    _transitions: TransitionsMapping[TState, TAction] = {
        "A": {
            "R": (("B", -1.0, 0.75), ("C", 1.0, 0.25)),
            "L": (("C", 1.0, 0.75), ("B", -1.0, 0.25)),
        },
        "B": {
            "R": (("C", 1.0, 0.75), ("A", -1.0, 0.25)),
            "L": (("A", -1.0, 0.75), ("C", 1.0, 0.25)),
        },
        "C": {"R": (("C", 0.0, 1.0),), "L": (("C", 0.0, 1.0),)},
    }

    @property
    def states(self) -> List[TState]:
        return self._states

    @property
    def actions(self) -> List[TAction]:
        return self._actions

    @property
    def rewards(self) -> Set[float]:
        return self._rewards

    def next_states_and_rewards(
        self, state: TState, action: TAction
    ) -> Collection[Tuple[TState, float, float]]:
        return self._transitions[state][action]


@pytest.fixture
def test_mdp() -> SimpleMDP:
    return SimpleMDP()


@pytest.fixture
def gamma() -> float:
    return 0.9


@pytest.fixture
def v() -> Dict[TState, float]:
    """Initial state values for use in tests."""
    return {"A": 3.0, "B": 1.0, "C": 0.0}


@pytest.fixture
def pi() -> Policy[TState, TAction]:
    """Stochastic policy for use in tests."""
    pi_dict = {
        "A": (("L", 0.6), ("R", 0.4)),
        "B": (("L", 1.0),),
        "C": (),
    }
    return lambda s: pi_dict[s]


class TestFiniteMDP:
    @pytest.mark.parametrize("state,index", [("A", 0), ("B", 1), ("C", 2)])
    def test_s2i(self, test_mdp: SimpleMDP, state: TState, index: int):
        assert test_mdp.s2i(state) == index

    @pytest.mark.parametrize("state,index", [("A", 0), ("B", 1), ("C", 2)])
    def test_i2s(self, test_mdp: SimpleMDP, state: TState, index: int):
        assert test_mdp.i2s(index) == state

    @pytest.mark.parametrize(
        "state,expected_v",
        [
            ("A", 0.55 * (1.0 + 0.0) + 0.45 * (-1.0 + 0.9 * 1.0)),
            ("B", 0.75 * (-1.0 + 0.9 * 3.0) + 0.25 * 1.0),
            ("C", 0.0),
        ],
    )
    def test_backup_single_state_value(
        self,
        test_mdp: SimpleMDP,
        v: Dict[TState, float],
        pi: Policy[TState, TAction],
        gamma: float,
        state: TState,
        expected_v: float,
    ) -> None:
        updated_v = test_mdp.backup_single_state_value(state, v, gamma, pi)
        assert_almost_equal(updated_v, expected_v)

    @pytest.mark.xfail
    def test_backup_single_state_optimal_action(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_backup_policy_values_operator(self) -> None:
        raise NotImplementedError

    @pytest.mark.xfail
    def test_backup_optimal_values(self) -> None:
        raise NotImplementedError
