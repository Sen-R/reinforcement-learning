from abc import ABC, abstractmethod
from typing import Generic, Sequence, Tuple, Callable, Mapping, Optional
from numpy.typing import NDArray
import numpy as np
from ._types import State, Action


class FiniteMDP(ABC, Generic[Action, State]):
    """Abstract base class for finite Markov Decision Processes."""

    @property
    @abstractmethod
    def states(self) -> Sequence[State]:
        """Returns list of possible states."""
        pass

    @property
    @abstractmethod
    def actions(self) -> Sequence[Action]:
        """Returns list of all possible actions."""
        pass

    @property
    @abstractmethod
    def rewards(self) -> Sequence[float]:
        """Returns list of all possible (expected) rewards."""
        pass

    @abstractmethod
    def s2i(self, state) -> int:
        """Converts a state to its corresponding index in `self.states`."""
        pass

    @abstractmethod
    def i2s(self, index) -> State:
        """Converts a state index to its corresponding state."""
        pass

    @abstractmethod
    def next_states_and_rewards(
        self, state: State, action: Action
    ) -> Sequence[Tuple[State, float, float]]:
        """Returns the next states, expected rewards and corresponding
        probabilities after taking `action` in `state`.

        Returns:
          A sequence of tuples `(ns, r, p_ns)`. Each element represents
          a possible successor state (`ns`), accompanying expected reward,
          (`r`) and the probability of ending up in that successor state
          (`p_ns`)."""
        pass

    def backup_single_state_value(
        self,
        state: State,
        v: Mapping[State, float],
        gamma: float,
        pi: Callable[[Action, State], float],
    ) -> float:
        """Updates estimated value of `state` from estimated value of
        successor states under policy `pi`.

        Args:
          state: state whose value is to be updated
          v: current mapping from states to values that is to be updated
          gamma: discount factor
          pi: policy encoded as conditional probabilities of actions given
            states

        Returns:
          updated value estimate for `state`
        """
        backed_up_v = 0.0
        for a in self.actions:
            for ns, r, p_ns in self.next_states_and_rewards(state, a):
                backed_up_v += pi(a, state) * p_ns * (r + gamma * v[ns])
        return backed_up_v

    def backup_single_state_optimal_action(
        self,
        state: State,
        v: Mapping[State, float],
        gamma: float,
    ) -> Tuple[Action, float]:
        """Returns an action and corresponding value that maximises expected
        return from `state`, estimated using the current state value mapping
        `v`.

        Args:
          state: current state, for which optimal action is estimated
          v: estimated state values, used to back-up optimal action
          gamma: discount factor

        Returns:
          action: maximising action, chosen arbitrarily if there are ties
          action_value: corresponding maximising action value
        """
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
        self,
        gamma: float,
        pi: Callable[[Action, State], float],
    ) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
        """Returns the matrix and vector components of the Bellman policy
        evaluation operator for this MDP.

        Args:
          gamma: discount factor
          pi: function that returns the probability of the given action being
            taken in the given state, according to the agent's policy

        Returns:
          A: matrix component of the Bellman operator, `gamma *t(s, s')`
            where `t` is the state transition matrix under the given policy
          b: vector component of the Bellman operator, i.e. the expected
            reward given state `s` (marginalising over all possible actions
            and transitioned states)
        """
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
        """Single update of the state-value function; RHS of Bellman
        optimality equation.

        Args:
          initial_values: array of initial state values to back-up
          gamma: discount factor

        Returns:
          array of updated values
        """
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
