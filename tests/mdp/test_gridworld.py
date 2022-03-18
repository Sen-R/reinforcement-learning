import pytest
from numpy.testing import assert_almost_equal
import numpy as np
from rl.mdp._types import Reward
from rl.mdp.gridworld import GridWorld, GWState, GWAction


class TestGridWorld:
    @pytest.mark.parametrize(
        "state,action,expected_next_state,expected_reward",
        [
            (GWState((2, 1)), GWAction("e"), GWState((2, 2)), 0),
            (GWState((3, 2)), GWAction("w"), GWState((3, 1)), 0),
            (GWState((2, 4)), GWAction("n"), GWState((1, 4)), 0),
            (GWState((1, 2)), GWAction("s"), GWState((2, 2)), 0),
            (GWState((0, 2)), GWAction("n"), GWState((0, 2)), -1),
            (GWState((4, 1)), GWAction("s"), GWState((4, 1)), -1),
            (GWState((2, 4)), GWAction("e"), GWState((2, 4)), -1),
            (GWState((1, 0)), GWAction("w"), GWState((1, 0)), -1),
            (GWState((0, 1)), GWAction("n"), GWState((4, 1)), 10),
            (GWState((0, 3)), GWAction("e"), GWState((2, 3)), 5),
        ],
    )
    def test_next_states_and_rewards(
        self,
        gridworld: GridWorld,
        state: GWState,
        action: GWAction,
        expected_next_state: GWState,
        expected_reward: Reward,
    ) -> None:
        """Tests whether `next_states_and_rewards` method delivers expected
        next states, rewards and probabilities."""
        assert gridworld.next_states_and_rewards(state, action) == (
            (
                (expected_next_state,),
                (1.0,),
            ),
            expected_reward,
        )

    @pytest.mark.parametrize(
        "state", [GWState((3, 0)), GWState((1, 4)), GWState((0, 0))]
    )
    def test_s2i(self, gridworld: GridWorld, state: GWState) -> None:
        assert gridworld.states[gridworld.s2i(state)] == state

    @pytest.mark.parametrize("idx", [2, 14, 22])
    def test_i2s(self, gridworld: GridWorld, idx: int) -> None:
        assert gridworld.i2s(idx) == gridworld.states[idx]

    def test_backup_single_state_value(self, gridworld: GridWorld) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = {s: gridworld.s2i(s) for s in gridworld.states}

        # Perform a backup update on a single state with arbitrary random
        # policy and check it meets expectations
        fixed_pi = (
            (GWAction("n"), 0.85),
            (GWAction("e"), 0.05),
            (GWAction("w"), 0.05),
            (GWAction("s"), 0.05),
        )
        updated_v = gridworld.backup_single_state_value(
            GWState((4, 0)), v, gamma=0.9, pi=lambda s: fixed_pi
        )
        assert gridworld.s2i(GWState((4, 0))) == 20  # sanity check
        expected_v = (
            0.05 * (-1 + 0.9 * 20)  # moving west
            + 0.85 * 0.9 * 15  # moving north
            + 0.05 * 0.9 * 21  # moving east
            + 0.05 * (-1 + 0.9 * 20)  # moving south
        )
        assert_almost_equal(updated_v, expected_v)

    def test_backup_single_state_optimal_action(
        self, gridworld: GridWorld
    ) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = {s: gridworld.s2i(s) for s in gridworld.states}

        # Call method to identify optimal action and action value estimated
        # for an arbitrarily chosen state and make sure it conforms to
        # expectations
        state = GWState((4, 0))
        assert gridworld.s2i(state) == 20  # sanity check
        actions, action_value = gridworld.backup_single_state_optimal_actions(
            state, v, gamma=0.9
        )
        assert len(actions) == 1
        assert actions[0] == GWAction("e")  # RHS worked out by hand
        assert action_value == 0.9 * 21  # corresponding action value

    def test_backup_policy_values_operator(self, gridworld: GridWorld) -> None:
        A, b = gridworld.backup_policy_values_operator(
            0.9,
            lambda s: [(a, 0.25) for a in gridworld.actions(s)],
        )
        assert b[gridworld.s2i(GWState((1, 2)))] == 0
        assert b[gridworld.s2i(GWState((2, 4)))] == -0.25
        assert b[gridworld.s2i(GWState((0, 3)))] == 5
        assert (
            A[gridworld.s2i(GWState((2, 4))), gridworld.s2i(GWState((2, 4)))]
            == 0.9 * 0.25
        )
        assert (
            A[gridworld.s2i(GWState((1, 2))), gridworld.s2i(GWState((3, 3)))]
            == 0
        )

    def test_backup_optimal_values(self, gridworld: GridWorld) -> None:
        initial_state_values = np.arange(len(gridworld.states), dtype=float)
        backed_up_values = gridworld.backup_optimal_values(
            initial_state_values, 0.9
        )

        # Check values match manual calculation
        assert backed_up_values[7] == 0.9 * 12
        assert backed_up_values[1] == 10 + 0.9 * 21

    def test_terminal_state_functionality(self) -> None:
        gridworld = GridWorld(2, terminal_states=(GWState((1, 0)),))
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("n")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 0))
        assert r == 0.0
        assert p_ns[0] == 1.0

    def test_wormholes_set_to_none_means_no_wormholes(self) -> None:
        gridworld = GridWorld(2)
        assert len(gridworld.wormholes) == 0

    def test_terminal_state_set_to_none_means_no_terminal_states(self) -> None:
        gridworld = GridWorld(2)
        assert len(gridworld.terminal_states) == 0

    def test_custom_move_reward(self) -> None:
        gridworld = GridWorld(2, default_move_reward=3.0)
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("e")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 1))
        assert r == 3.0
        assert p_ns[0] == 1.0

    def test_custom_invalid_action_reward(self) -> None:
        gridworld = GridWorld(2, invalid_action_reward=5.0)
        (ns, p_ns), r = gridworld.next_states_and_rewards(
            GWState((1, 0)), GWAction("w")
        )
        assert len(ns) == 1
        assert len(p_ns) == 1
        assert ns[0] == GWState((1, 0))
        assert r == 5.0
        assert p_ns[0] == 1.0
