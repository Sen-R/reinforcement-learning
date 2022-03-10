from rl.mdp.jackscarrental import JacksCarRental
from rl.mdp.solve import value_iteration


# Jack's car rental problem with parameters specified in Sutton, Barto
jcr = JacksCarRental(
    capacity=20,
    exp_demand_per_location=(1, 2),
    exp_returns_per_location=(1, 1),
    reward_for_rental=10.0,
    reward_per_car_for_moving_cars=-2.0,
)


if __name__ == "__main__":
    v = {s: 0.0 for s in jcr.states}
    value_iteration(v, jcr, 0.9, 1.0, maxiter=1)
    print(v)
