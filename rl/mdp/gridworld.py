from typing import (
    NewType,
    Tuple,
    Sequence,
    Callable,
    Mapping,
    Optional,
)
from itertools import product
import numpy as np
from numpy.typing import NDArray
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

    def backup_single_state_value(
        self,
        state: State,
        v: Mapping[State, float],
        gamma: float,
        pi: Callable[[Action, State], float],
    ) -> float:
        backed_up_v = 0.0
        for a in self.actions:
            for ns, r, p_ns in self.next_states_and_rewards(state, a):
                backed_up_v += pi(a, state) * p_ns * (r + gamma * v[ns])
        return backed_up_v

    def backup_single_state_optimal_action(
        self, state: State, v: Mapping[State, float], gamma: float
    ) -> Tuple[Action, float]:
        best_action_and_value: Optional[Tuple[Action, float]] = None
        for a in self.actions:
            this_action_value = sum(
                p_ns * (r + gamma * v[ns])
                for ns, r, p_ns in self.next_states_and_rewards(state, a)
            )
            if (
                best_action_and_value is None
                or this_action_value > best_action_and_value[1]
            ):
                best_action_and_value = (a, this_action_value)
        assert best_action_and_value is not None
        return best_action_and_value

    def backup_policy_values_operator(
        self, gamma: float, pi: Callable[[Action, State], float]
    ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        expected_rewards_vector = np.zeros(len(self.states))
        discounted_transitions_matrix = np.zeros(
            (len(self.states), len(self.states))
        )

        for s in self.states:
            for a in self.actions:
                p_a = pi(a, s)
                for ns, r, p_ns in self.next_states_and_rewards(s, a):
                    expected_rewards_vector[self.s2i(s)] += p_a * p_ns * r
                    discounted_transitions_matrix[
                        self.s2i(s), self.s2i(ns)
                    ] += (gamma * p_a * p_ns)

        return discounted_transitions_matrix, expected_rewards_vector

    def backup_optimal_values(
        self, initial_values: NDArray[np.float_], gamma: float
    ) -> NDArray[np.float_]:
        initial_values = np.array(initial_values)
        updated_values = np.zeros(len(self.states))
        for s in self.states:
            updated_values[self.s2i(s)] = max(
                sum(
                    p_ns * (r + gamma * initial_values[self.s2i(ns)])
                    for ns, r, p_ns in self.next_states_and_rewards(s, a)
                )
                for a in self.actions
            )
        return updated_values
