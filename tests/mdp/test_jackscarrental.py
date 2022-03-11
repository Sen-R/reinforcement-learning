from typing import Tuple, Collection, Iterable
import pytest
from numpy.testing import assert_almost_equal
from scipy.stats import poisson  # type: ignore
from rl.mdp.jackscarrental import (
    JacksCarRental,
    CarCounts,
    MoveCars,
    counts_after_moving_cars,
)


@pytest.fixture
def jcr() -> JacksCarRental:
    """Returns a MDP for the Jack's car rental problem, as specified in
    Sutton, Barto (2018) section 4.3, but with changed parameters to keep
    the state space size manageable for testing purposes."""
    mdp = JacksCarRental(
        capacity=2,
        exp_demand_per_location=(1, 2),
        exp_returns_per_location=(1, 1),
        reward_for_rental=10.0,
        reward_per_car_for_moving_cars=-2.0,
    )
    return mdp


p_rent = (poisson(mu=1), poisson(mu=2))
p_ret = (poisson(mu=1), poisson(mu=1))


class TestJacksCarRental:
    def test_states_property(self, jcr: JacksCarRental) -> None:
        """States should consist of a tuple of ints, each between 1 and 20,
        representing the number of cars at the first and second location
        respectively."""
        expected_states = {
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        }
        actual_states = set(jcr.states)
        assert expected_states == actual_states

    @pytest.mark.parametrize(
        "state,exp_range",
        [
            (CarCounts((1, 2)), range(-1, 1)),
            (CarCounts((2, 0)), range(0, 3)),
            (CarCounts((1, 1)), range(-1, 2)),
            (CarCounts((0, 2)), range(-2, 1)),
            (CarCounts((0, 0)), range(0, 1)),
            (CarCounts((2, 2)), range(0, 1)),
        ],
    )
    def test_actions_property(
        self, jcr: JacksCarRental, state: CarCounts, exp_range: Iterable[int]
    ) -> None:
        """Actions should be a (possibly negative) int, representing
        the net movement of cars from location 1 to location 2 overnight."""
        expected_actions = set(MoveCars(m) for m in exp_range)
        actual_actions = set(jcr.actions(state))
        assert expected_actions == actual_actions

    def test_next_states_and_rewards(self, jcr: JacksCarRental) -> None:
        state = CarCounts((1, 2))  # arbitrary state
        action = MoveCars(-1)  # arbitrary action
        morning_counts = CarCounts((2, 1))  # after applying action
        evening_counts_and_rentals = jcr.evening_states_and_exp_rentals(
            morning_counts
        )
        exp_ns_rs_and_ps = [
            (ns, r * 10.0 - 2.0, p) for ns, r, p in evening_counts_and_rentals
        ]
        act_ns_rs_and_ps = jcr.next_states_and_rewards(state, action)
        for (act_ns, act_r, act_p), (exp_ns, exp_r, exp_p) in zip(
            act_ns_rs_and_ps, exp_ns_rs_and_ps
        ):
            assert act_ns == exp_ns
            print(act_ns)
            assert_almost_equal(act_r, exp_r)
            assert_almost_equal(act_p, exp_p)

    def test_evening_states_and_exp_rentals(self, jcr: JacksCarRental) -> None:
        morning_counts = CarCounts((1, 2))
        ns_rs_and_ps = jcr.evening_states_and_exp_rentals(morning_counts)

        # All states are possible next states, let's check this first
        assert set(ns for ns, _, _ in ns_rs_and_ps) == set(jcr.states)

        # Probabilities should sum to one
        assert_almost_equal(sum(p for _, _, p in ns_rs_and_ps), 1.0)

    @pytest.mark.parametrize(
        "cars_morning,cars_evening,rs_and_ps",
        [
            (
                1,
                1,
                (
                    # same transition but for branch 1
                    (0, p_rent[1].pmf(0) * p_ret[1].pmf(0)),
                    (1, (1.0 - p_rent[1].cdf(0)) * p_ret[1].pmf(1)),
                ),
            ),
            (
                2,
                2,
                (
                    # logic different if branch ends up full
                    (0, p_rent[1].pmf(0)),
                    (1, p_rent[1].pmf(1) * (1.0 - p_ret[1].cdf(0))),
                    (2, (1.0 - p_rent[1].cdf(1)) * (1.0 - p_ret[1].cdf(1))),
                ),
            ),
        ],
    )
    def test_branch_transition_prob_and_exp_rentals(
        self,
        jcr: JacksCarRental,
        cars_morning: int,
        cars_evening: int,
        rs_and_ps: Collection[Tuple[int, float]],
    ) -> None:
        expected_p = sum(r_and_p[1] for r_and_p in rs_and_ps)
        expected_r = (
            sum(r_and_p[0] * r_and_p[1] for r_and_p in rs_and_ps) / expected_p
        )
        actual_p, actual_r = jcr.branch_transition_prob_and_exp_rentals(
            1, cars_morning=cars_morning, cars_evening=cars_evening
        )
        assert_almost_equal(actual_p, expected_p)
        assert_almost_equal(actual_r, expected_r)


@pytest.mark.parametrize(
    "state,action,counts_after_move",
    [
        (CarCounts((1, 2)), MoveCars(-1), CarCounts((2, 1))),
        (CarCounts((1, 1)), MoveCars(1), CarCounts((0, 2))),
    ],
)
def test_counts_after_moving_cars(
    state: CarCounts,
    action: MoveCars,
    counts_after_move: CarCounts,
) -> None:
    assert counts_after_moving_cars(state, action) == counts_after_move
