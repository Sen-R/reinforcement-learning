""" Implementation of Jack's car rental problem as a `FiniteMDP`.

This problem is described in chapter 4 of the textbook Sutton, Barto (2018).
"""

from typing import NewType, Tuple, List, Dict
from itertools import product
from numpy.typing import NDArray
import numpy as np
from scipy.stats import poisson  # type: ignore
from .base import FiniteMDP
from ._types import NextStateRewardAndProbability

CarCounts = NewType("CarCounts", Tuple[int, int])
MoveCars = NewType("MoveCars", int)


class JacksCarRental(FiniteMDP[CarCounts, MoveCars]):
    """MDP for Jack's car rental.

    Args:
      capacity: number of cars that can be held at a location
      exp_demand_per_location: mean daily rental demand at each location
      exp_returns_per_location: mean daily returns at each location
      reward_for_rental: dollar credit received for renting a car out
        (this is $10 in the problem as originally stated)
      reward_per_car_for_moving_cars: (negative) dollar reward for each
        car to be moved from one location to the other overnight (this is
        -$2 in the problem as originally stated)
    """

    def __init__(
        self,
        capacity: int,
        exp_demand_per_location: Tuple[int, int],
        exp_returns_per_location: Tuple[int, int],
        reward_for_rental: float,
        reward_per_car_for_moving_cars: float,
    ):
        self.capacity = capacity
        self.exp_demand_per_location = exp_demand_per_location
        self.exp_returns_per_location = exp_returns_per_location
        self.reward_for_rental = reward_for_rental
        self.reward_per_car_for_moving_cars = reward_per_car_for_moving_cars
        self.p_hire = (
            poisson_table(self.capacity, exp_demand_per_location[0]),
            poisson_table(self.capacity, exp_demand_per_location[1]),
        )
        self.p_ret = (
            poisson_table(self.capacity, exp_returns_per_location[0]),
            poisson_table(self.capacity, exp_returns_per_location[1]),
        )
        self._cache: Dict[CarCounts, List[Tuple[CarCounts, float, float]]] = {}

    @property
    def states(self) -> List[CarCounts]:
        return [
            CarCounts(t)
            for t in product(
                range(self.capacity + 1), range(self.capacity + 1)
            )
        ]

    def actions(self, state: CarCounts) -> List[MoveCars]:
        min_move = -min(self.capacity - state[0], state[1])
        max_move = min(self.capacity - state[1], state[0])
        return [MoveCars(m) for m in range(min_move, max_move + 1)]

    def next_states_and_rewards(
        self, state: CarCounts, action: MoveCars
    ) -> List[NextStateRewardAndProbability[CarCounts]]:
        next_morning_counts = counts_after_moving_cars(state, action)
        assert max(next_morning_counts) <= self.capacity, (state, action)
        assert min(next_morning_counts) >= 0, (state, action)
        next_evening_counts_and_rentals = self.evening_states_and_exp_rentals(
            next_morning_counts
        )
        action_reward = abs(action) * self.reward_per_car_for_moving_cars
        return [
            (
                next_state,
                rentals * self.reward_for_rental + action_reward,
                prob,
            )
            for next_state, rentals, prob in next_evening_counts_and_rentals
        ]

    def evening_states_and_exp_rentals(
        self, morning_counts: CarCounts
    ) -> List[Tuple[CarCounts, float, float]]:
        """Returns possible car counts at the end of a day, expected number
        of cars rented over the day, and associated probabilities.

        Args:
          morning_counts: the number cars at each location at the start of the
            day

        Returns:
          A list of tuples `(evening_counts, cars_rented, prob)`, enumerating
          the possibilities for the end of that day: i.e. the possible end of
          day car counts per location (`evening_counts`), the associated
          expected number of cars rented over the day (`cars rented`), and
          the associated probability of this outcome occurring (`prob`),
          conditional on `morning_counts` cars being present at the start of
          the day.
        """
        try:
            return self._cache[morning_counts]
        except KeyError:
            result: List[Tuple[CarCounts, float, float]] = []
            for next_state in self.states:
                p0, r0 = self.branch_transition_prob_and_exp_rentals(
                    0, morning_counts[0], next_state[0]
                )
                p1, r1 = self.branch_transition_prob_and_exp_rentals(
                    1, morning_counts[1], next_state[1]
                )
                r = r0 + r1
                p = p0 * p1
                result.append((CarCounts(next_state), r, p))
            self._cache[morning_counts] = result
            return result

    def branch_transition_prob_and_exp_rentals(
        self, branch: int, cars_morning: int, cars_evening: int
    ) -> Tuple[float, float]:
        """Returns probability of going from `cars_morning` cars at the start
        of the day to `cars_evening` cars at the end, given branch
        characteristics. Also gives expected number of cars rented out given
        that transition.

        Note: exp_r will have a high level of numerical error when total_prob
        is small, but this doesn't matter because only the product of both
        is used. When total_prob is zero, exp_r is overriden to zero to prevent
        divide by zero warnings. In future, it would be good to improve the
        implementation though to avoid these issues in the first place.

        Args:
          branch: index of the branch for which this is being calculated
          cars_morning: number of cars at this branch at the start of the day
          cars_evening: number of cars at this branch at the end of the day

        Returns:
          total_prob: probability of having `cars_evening` cars at the end of
            the day, given the branch started the day with `cars_morning` cars
          exp_r: expected number of cars rented out over the day, given that
            the day started with `cars_morning` cars and ended with
            `cars_evening` cars
        """
        probs: List[float] = []
        rentals: List[float] = []
        for cars_hired in range(
            max(0, cars_morning - cars_evening), cars_morning + 1
        ):
            cars_returned = cars_evening - cars_morning + cars_hired
            assert cars_hired < len(self.p_hire[branch]), cars_hired
            assert cars_returned < len(self.p_ret[branch]), cars_returned
            assert cars_returned >= 0, (cars_hired, cars_returned)
            probs.append(
                (
                    # prob of `cars_hired` cars being rented out
                    self.p_hire[branch][cars_hired]
                    if cars_hired < cars_morning
                    else 1.0 - np.sum(self.p_hire[branch][:cars_hired])
                )
                * (
                    # prob of `cars_returned` cars being returned
                    self.p_ret[branch][cars_returned]
                    if cars_evening < self.capacity
                    else 1.0 - np.sum(self.p_ret[branch][:cars_returned])
                )
            )
            rentals.append(cars_hired)
        total_prob = sum(probs)
        if total_prob == 0.0:
            return 0.0, 0.0  # avoid divide by zero
        else:
            exp_r = sum(p * r for p, r in zip(probs, rentals)) / total_prob
            return total_prob, exp_r


def counts_after_moving_cars(
    initial_counts: CarCounts,
    move: MoveCars,
) -> CarCounts:
    """Returns car counts in the morning after `move` cars were moved
    overnight.

    Args:
      initial_counts: number of cars at each location prior to any overnight
      moves
      move: number of cars to move from location 0 to location 1 (negative
        number means moving cars in the opposite direction)

    Returns:
      Number of cars at each location the following morning, after overnight
      moves have taken place
    """
    return CarCounts((initial_counts[0] - move, initial_counts[1] + move))


def poisson_table(max_k: int, mu: float) -> NDArray[np.float_]:
    return poisson.pmf(range(max_k + 1), mu=mu)
    # table = np.zeros(max_k + 1)
    # table[:max_k] = poisson.pmf(range(max_k), mu=mu)
    # assert table[-1] == 0.0
    # table[-1] = 1.0 - poisson.cdf(max_k - 1, mu=mu)
    # np.testing.assert_almost_equal(np.sum(table), 1.0)
    # return table
