from typing import Collection
from rl.mdp.jackscarrental import JacksCarRental, CarCounts
from rl.mdp.solve import policy_iteration


# Jack's car rental problem with parameters specified in Sutton, Barto
jcr = JacksCarRental(
    capacity=20,
    exp_demand_per_location=(1, 2),
    exp_returns_per_location=(1, 1),
    reward_for_rental=10.0,
    reward_per_car_for_moving_cars=-2.0,
)


def print_row_vector(row: Collection[float], width: int, digits: int) -> None:
    template = f"{{:{width}.{digits}f}}"
    print(" ".join(template.format(el) for el in row))


def print_matrix(
    matrix: Collection[Collection[float]], width: int, digits: int
) -> None:
    for row in matrix:
        print_row_vector(row, width, digits)


if __name__ == "__main__":
    v = {s: 0.0 for s in jcr.states}
    pi = {s: jcr.actions(s)[0] for s in jcr.states}
    policy_iteration(v, pi, jcr, 0.9, 1.0)
    print("State values:")
    print_matrix(
        [[v[CarCounts((i, j))] for j in range(21)] for i in range(21)], 3, 0
    )
    print("\nOptimal action:")
    print_matrix(
        [[pi[CarCounts((i, j))] for j in range(21)] for i in range(21)], 2, 0
    )
