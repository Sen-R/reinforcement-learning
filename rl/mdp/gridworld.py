from typing import NewType, Tuple, Sequence, Mapping
from itertools import product
from .base import FiniteMDP


State = NewType("State", Tuple[int, int])
Action = NewType("Action", str)


class GridWorld(FiniteMDP[Action, State]):
    """A GridWorld finite MDP, generalising the example given in
    Chapter 3 of Sutton, Barto (2018)."""

    def __init__(
        self, size: int, wormholes: Mapping[State, Tuple[State, float]]
    ):
        self.size = size
        self.wormholes = wormholes
        self._states = [
            State((i, j)) for i, j in product(range(size), range(size))
        ]
        self.actions_to_moves = {
            "n": (-1, 0),
            "e": (0, 1),
            "w": (0, -1),
            "s": (1, 0),
        }

    @property
    def states(self) -> Sequence[State]:
        return self._states

    @property
    def actions(self) -> Sequence[Action]:
        return list(Action(a) for a in self.actions_to_moves)

    @property
    def rewards(self) -> Sequence[float]:
        return [0.0, -1.0] + [w[1] for w in self.wormholes.values()]

    def s2i(self, state: State) -> int:
        return state[0] * self.size + state[1]

    def i2s(self, index: int) -> State:
        return State((index // self.size, index % self.size))

    def next_states_and_rewards(
        self, state: State, action: Action
    ) -> Tuple[Tuple[State, float, float]]:
        if state in self.wormholes:
            return ((*self.wormholes[state], 1.0),)
        else:
            move = self.actions_to_moves[action]
            next_state = State((state[0] + move[0], state[1] + move[1]))
            if self.state_is_valid(next_state):
                return ((next_state, 0, 1.0),)
            else:
                return ((state, -1, 1.0),)

    def state_is_valid(self, state):
        return min(state) >= 0 and max(state) < self.size
