from typing import List, Collection, Tuple
import pytest
from rl.mdp._types import TransitionsMapping, NextStateProbabilityTable
from rl.mdp import FiniteMDP, GridWorld


@pytest.fixture
def gridworld():
    """Constructs the gridworld described in Chapter 3 of Sutton, Barto (2018)
    textbook.
    """
    return GridWorld(
        size=5,
        wormholes={
            (0, 1): ((4, 1), 10),
            (0, 3): ((2, 3), 5),
        },
    )


TState = str
TAction = str


class SimpleMDP(FiniteMDP[TState, TAction]):
    _states = ["A", "B", "C"]
    _actions = ["R", "L"]
    _transitions: TransitionsMapping[TState, TAction] = {
        "A": {
            "R": (("B", 0.75, -1.0), ("C", 0.25, 1.0)),
            "L": (("C", 0.75, 1.0), ("B", 0.25, -1.0)),
        },
        "B": {
            "R": (("C", 0.75, 1.0), ("A", 0.25, -1.0)),
            "L": (("A", 0.75, -1.0), ("C", 0.25, 1.0)),
        },
        "C": {"R": (("C", 1.0, 0.0),), "L": (("C", 1.0, 0.0),)},
    }

    @property
    def states(self) -> List[TState]:
        return self._states

    def actions(self, state: TState) -> List[TAction]:
        return self._actions

    def next_states_and_rewards(
        self, state: TState, action: TAction
    ) -> Tuple[NextStateProbabilityTable, float]:
        transitions: Collection[
            Tuple[TState, float, float]
        ] = self._transitions[state][action]
        ns_ptable = (
            [ns for ns, p, r in transitions],
            [p for ns, p, r in transitions],
        )
        exp_r = sum(p * r for ns, p, r in transitions)
        return ns_ptable, exp_r


@pytest.fixture
def test_mdp() -> SimpleMDP:
    return SimpleMDP()
