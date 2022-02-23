from typing import NewType, Tuple, Sequence, Callable, Mapping, MutableMapping
from itertools import product
import numpy as np
from numpy.typing import ArrayLike
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

    def next_state_and_reward(
        self, state: State, action: Action
    ) -> Tuple[State, float]:
        """Returns the next state and reward after taking `action` in
        `state`."""
        if state in self.wormholes:
            return self.wormholes[state]
        else:
            move = self.actions_to_moves[action]
            next_state = State((state[0] + move[0], state[1] + move[1]))
            if self.state_is_valid(next_state):
                return next_state, 0
            else:
                return state, -1

    def state_is_valid(self, state):
        return min(state) >= 0 and max(state) < self.size

    def backup_single_state_value(
        self,
        state: State,
        v: MutableMapping[State, float],
        gamma: float,
        pi: Callable[[Action, State], float],
    ):
        backed_up_v = 0.0
        for action in self.actions:
            next_state, reward = self.next_state_and_reward(state, action)
            backed_up_v += pi(action, state) * (reward + gamma * v[next_state])
        v[state] = backed_up_v

    def backup_policy_values_operator(
        self, gamma: float, pi: Callable[[Action, State], float]
    ) -> Tuple[ArrayLike, ArrayLike]:
        expected_rewards_vector = np.zeros(len(self.states))
        discounted_transitions_matrix = np.zeros(
            (len(self.states), len(self.states))
        )

        for state in self.states:
            for action in self.actions:
                next_state, reward = self.next_state_and_reward(state, action)
                action_probability = pi(action, state)
                expected_rewards_vector[self.s2i(state)] += (
                    action_probability * reward
                )
                discounted_transitions_matrix[
                    self.s2i(state), self.s2i(next_state)
                ] += (gamma * action_probability)

        return discounted_transitions_matrix, expected_rewards_vector

    def backup_optimal_values(
        self, initial_values: ArrayLike, gamma: float
    ) -> ArrayLike:
        updated_values = np.zeros(len(self.states))
        for state in self.states:
            action_values = []
            for action in self.actions:
                next_state, reward = self.next_state_and_reward(state, action)
                action_values.append(
                    reward
                    + gamma * np.array(initial_values)[self.s2i(next_state)]
                )
            updated_values[self.s2i(state)] = max(action_values)
        return updated_values
