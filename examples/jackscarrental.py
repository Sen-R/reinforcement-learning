from typing import Collection, Mapping
from numpy.testing import assert_almost_equal
from rl.mdp.jackscarrental import JacksCarRental, CarCounts
from rl.mdp.solve import (
    policy_iteration,
    value_iteration,
    optimal_actions_from_state_values,
)


# Jack's car rental problem with parameters specified in Sutton, Barto
jcr = JacksCarRental(
    capacity=20,
    overnight_moves_limit=5,
    exp_demand_per_location=(3, 4),
    exp_returns_per_location=(3, 2),
    reward_for_rental=10.0,
    reward_per_car_for_moving_cars=-2.0,
)


def print_row_vector(row: Collection[float], width: int, digits: int) -> None:
    template = f"{{:{width}.{digits}f}}"
    print(" ".join(template.format(el) for el in row))


def plot_2d_function(
    mapping: Mapping[CarCounts, float], width: int, digits: int
) -> None:
    x_vals, y_vals = tuple(zip(*mapping.keys()))
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    for x in range(x_max, x_min - 1, -1):
        print_row_vector(
            [mapping[CarCounts((x, y))] for y in range(y_min, y_max + 1)],
            width,
            digits,
        )


if __name__ == "__main__":
    v_pol_it = {s: 0.0 for s in jcr.states}
    pi = {s: jcr.actions(s)[0] for s in jcr.states}
    policy_iteration(v_pol_it, pi, jcr, 0.9, 0.1)
    print("===== Policy Iteration =====")
    print("State values:")
    plot_2d_function(v_pol_it, 3, 0)
    print("\nOptimal action:")
    plot_2d_function(pi, 3, 0)

    # Corroborate against results from value iteration
    v_val_it = {s: 0.0 for s in jcr.states}
    value_iteration(v_val_it, jcr, 0.9, 0.001, maxiter=100)
    for s in jcr.states:
        assert_almost_equal(
            v_pol_it[s], v_val_it[s], err_msg=f"State: {s}", decimal=0
        )
    optimal_actions_map = optimal_actions_from_state_values(jcr, v_val_it, 0.9)
    for state, actions_list in optimal_actions_map.items():
        assert len(actions_list) > 0, (state, actions_list)
        assert pi[state] in actions_list, (state, pi[state], actions_list)
