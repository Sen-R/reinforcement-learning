import pytest
from rl.mdp import GridWorld


@pytest.fixture
def gridworld():
    """Constructs the gridworld described in Chapter 3 of Sutton, Barto (2018)
    textbook.
    """
    return GridWorld(
        size=5,
        wormholes={
            (0, 1): ((4, 1), 10),
            (0, 3): ((2, 3), 5),
        },
    )
