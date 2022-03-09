""" Implementation of Jack's car rental problem as a `FiniteMDP`.

This problem is described in chapter 4 of the textbook Sutton, Barto (2018).
"""

from typing import NewType, Tuple, List
from itertools import product
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

    @property
    def states(self) -> List[CarCounts]:
        return [
            CarCounts(t)
            for t in product(
                range(self.capacity + 1), range(self.capacity + 1)
            )
        ]

    def actions(self, state: CarCounts) -> List[MoveCars]:
        return [MoveCars(m) for m in range(-state[1], state[0] + 1)]

    def next_states_and_rewards(
        self, state: CarCounts, action: MoveCars
    ) -> List[NextStateRewardAndProbability[CarCounts]]:
        next_morning_counts = counts_after_moving_cars(state, action)
        next_evening_counts_and_rentals = evening_states_and_exp_rentals(
            next_morning_counts,
            self.capacity,
            self.exp_demand_per_location,
            self.exp_returns_per_location,
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


def evening_states_and_exp_rentals(
    morning_counts: CarCounts,
    capacity: int,
    exp_demand_per_location: Tuple[int, int],
    exp_returns_per_location: Tuple[int, int],
) -> List[Tuple[CarCounts, float, float]]:
    """Returns possible car counts at the end of a day, expected number
    of cars rented over the day, and associated probabilities.

    Args:
      morning_counts: the number cars at each location at the start of the
        day
      capacity: the number of cars that can be held at each location
      exp_demand_per_location: mean daily number of cars rented at each
        location
      exp_returns_per_location: mean daily returns at each location

    Returns:
      A list of tuples `(evening_counts, cars_rented, prob)`, enumerating
      the possibilities for the end of that day: i.e. the possible end of
      day car counts per location (`evening_counts`), the associated
      expected number of cars rented over the day (`cars rented`), and
      the associated probability of this outcome occurring (`prob`),
      conditional on `morning_counts` cars being present at the start of
      the day.
    """
    result: List[Tuple[CarCounts, float, float]] = []
    state_space = product(range(capacity + 1), range(capacity + 1))
    for next_state in state_space:
        p0, r0 = branch_transition_prob_and_exp_rentals(
            morning_counts[0],
            next_state[0],
            capacity,
            exp_demand_per_location[0],
            exp_returns_per_location[0],
        )
        p1, r1 = branch_transition_prob_and_exp_rentals(
            morning_counts[1],
            next_state[1],
            capacity,
            exp_demand_per_location[1],
            exp_returns_per_location[1],
        )
        r = r0 + r1
        p = p0 * p1
        result.append((CarCounts(next_state), r, p))
    return result


def branch_transition_prob_and_exp_rentals(
    cars_morning: int,
    cars_evening: int,
    capacity: int,
    exp_demand: int,
    exp_returns: int,
) -> Tuple[float, float]:
    """Returns probability of going from `cars_morning` cars at the start
    of the day to `cars_evening` cars at the end, given branch
    characteristics. Also gives expected number of cars rented out given
    that transition.

    Args:
      cars_morning: number of cars at this branch at the start of the day
      cars_evening: number of cars at this branch at the end of the day
      capacity: number of cars that can be held at this branch
      exp_demand: mean daily number of cars rented out at this branch
      exp_return: mean daily number of returns at this branch

    Returns:
      total_prob: probability of having `cars_evening` cars at the end of
        the day, given the branch started the day with `cars_morning` cars
      exp_r: expected number of cars rented out over the day, given that
        the day started with `cars_morning` cars and ended with
        `cars_evening` cars
    """
    p_hire = poisson(mu=exp_demand)
    p_ret = poisson(mu=exp_returns)
    probs: List[float] = []
    rentals: List[float] = []
    for cars_hired in range(cars_morning + 1):
        cars_returned = cars_evening - cars_morning + cars_hired
        probs.append(
            (
                # prob of `cars_hired` cars being rented out
                p_hire.pmf(cars_hired)
                if cars_hired < cars_morning
                else 1.0 - p_hire.cdf(cars_hired - 1)
            )
            * (
                # prob of `cars_returned` cars being returned
                p_ret.pmf(cars_returned)
                if cars_evening < capacity
                else 1.0 - p_ret.cdf(cars_returned - 1)
            )
        )
        rentals.append(cars_hired)
    total_prob = sum(probs)
    exp_r = sum(p * r for p, r in zip(probs, rentals)) / total_prob
    return total_prob, exp_r
