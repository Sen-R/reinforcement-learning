import pytest
from numpy.testing import assert_almost_equal


class TestGridWorld:
    @pytest.mark.parametrize(
        "state,action,expected_next_state,expected_reward",
        [
            ((2, 1), "e", (2, 2), 0),
            ((3, 2), "w", (3, 1), 0),
            ((2, 4), "n", (1, 4), 0),
            ((1, 2), "s", (2, 2), 0),
            ((0, 2), "n", (0, 2), -1),
            ((4, 1), "s", (4, 1), -1),
            ((2, 4), "e", (2, 4), -1),
            ((1, 0), "w", (1, 0), -1),
            ((0, 1), "n", (4, 1), 10),
            ((0, 3), "e", (2, 3), 5),
        ],
    )
    def test_next_states_and_rewards(
        self,
        gridworld,
        state,
        action,
        expected_next_state,
        expected_reward,
    ) -> None:
        """Tests whether `next_states_and_rewards` method delivers expected
        next states, rewards and probabilities."""
        assert gridworld.next_states_and_rewards(state, action) == (
            (expected_next_state, expected_reward, 1.0),
        )

    def test_rewards_property(self, gridworld) -> None:
        assert sorted(gridworld.rewards) == [-1, 0, 5, 10]

    @pytest.mark.parametrize("state", [(3, 0), (1, 4), (0, 0)])
    def test_s2i(self, gridworld, state) -> None:
        assert gridworld.states[gridworld.s2i(state)] == state

    @pytest.mark.parametrize("idx", [2, 14, 22])
    def test_i2s(self, gridworld, idx) -> None:
        assert gridworld.i2s(idx) == gridworld.states[idx]

    def test_backup_single_state_value(self, gridworld) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = {s: gridworld.s2i(s) for s in gridworld.states}

        # Perform a backup update on a single state with arbitrary random
        # policy and check it meets expectations
        fixed_pi = (("n", 0.85), ("e", 0.05), ("w", 0.05), ("s", 0.05))
        updated_v = gridworld.backup_single_state_value(
            (4, 0), v, gamma=0.9, pi=lambda s: fixed_pi
        )
        assert gridworld.s2i((4, 0)) == 20  # sanity check
        expected_v = (
            0.05 * (-1 + 0.9 * 20)  # moving west
            + 0.85 * 0.9 * 15  # moving north
            + 0.05 * 0.9 * 21  # moving east
            + 0.05 * (-1 + 0.9 * 20)  # moving south
        )
        assert_almost_equal(updated_v, expected_v)

    def test_backup_single_state_optimal_action(self, gridworld) -> None:
        # Initialise state value mapping arbitrarily as follows
        v = {s: gridworld.s2i(s) for s in gridworld.states}

        # Call method to identify optimal action and action value estimated
        # for an arbitrarily chosen state and make sure it conforms to
        # expectations
        state = (4, 0)
        assert gridworld.s2i(state) == 20  # sanity check
        action, action_value = gridworld.backup_single_state_optimal_action(
            state, v, gamma=0.9
        )
        assert action == "e"  # worked out by hand this is optimal
        assert action_value == 0.9 * 21  # corresponding action value

    def test_backup_policy_values_operator(self, gridworld) -> None:
        A, b = gridworld.backup_policy_values_operator(
            0.9,
            lambda s: [(a, 0.25) for a in gridworld.actions],
        )
        assert b[gridworld.s2i((1, 2))] == 0
        assert b[gridworld.s2i((2, 4))] == -0.25
        assert b[gridworld.s2i((0, 3))] == 5
        assert A[gridworld.s2i((2, 4)), gridworld.s2i((2, 4))] == 0.9 * 0.25
        assert A[gridworld.s2i((1, 2)), gridworld.s2i((3, 3))] == 0

    def test_backup_optimal_values(self, gridworld) -> None:
        initial_state_values = range(len(gridworld.states))
        backed_up_values = gridworld.backup_optimal_values(
            initial_state_values, 0.9
        )

        # Check values match manual calculation
        assert backed_up_values[7] == 0.9 * 12
        assert backed_up_values[1] == 10 + 0.9 * 21
