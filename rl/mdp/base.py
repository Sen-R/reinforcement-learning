from typing import Generic, Sequence, Tuple, Callable, MutableMapping
from numpy.typing import ArrayLike
from ._types import State, Action
from abc import ABC, abstractmethod


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
    def backup_single_state_value(
        self,
        state: State,
        v: MutableMapping[State, float],
        gamma: float,
        pi: Callable[[Action, State], float],
    ) -> None:
        """Updates (in place) estimated value of `state` from estimated value of
        successor states under policy `pi`.

        Args:
          state: state whose value is to be updated
          v: mapping from states to values that is to be updated (in place)
          gamma: discount factor
          pi: policy encoded as conditional probabilities of actions given
            states
        """
        pass

    @abstractmethod
    def backup_single_state_optimal_action(
        self,
        state: State,
        v: MutableMapping[State, float],
        gamma: float,
    ) -> Action:
        """Returns an action that maximises expected return from `state`,
        estimated using the current state value mapping `v`.

        Args:
          state: current state, for which optimal action is estimated
          v: estimated state values, used to back-up optimal action
          gamma: discount factor

        Returns:
          Maximising action, chosen arbitrarily if there are ties
        """
        pass

    @abstractmethod
    def backup_policy_values_operator(
        self,
        gamma: float,
        pi: Callable[[Action, State], float],
    ) -> Tuple[ArrayLike, ArrayLike]:
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
        pass

    @abstractmethod
    def backup_optimal_values(
        self, initial_values: ArrayLike, gamma: float
    ) -> ArrayLike:
        """Single update of the state-value function; RHS of Bellman
        optimality equation.

        Args:
          initial_values: array of initial state values to back-up
          gamma: discount factor

        Returns:
          array of updated values
        """
        pass
