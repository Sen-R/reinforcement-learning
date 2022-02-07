import pytest


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
    def test_next_state_and_reward(
        self,
        gridworld,
        state,
        action,
        expected_next_state,
        expected_reward,
    ) -> None:
        """Tests whether `next_state_and_reward` method delivers expected
        next states and rewards."""
        assert gridworld.next_state_and_reward(state, action) == (
            expected_next_state,
            expected_reward,
        )

    def test_rewards_property(self, gridworld) -> None:
        assert sorted(gridworld.rewards) == [-1, 0, 5, 10]

    @pytest.mark.parametrize("state", [(3, 0), (1, 4), (0, 0)])
    def test_s2i(self, gridworld, state) -> None:
        assert gridworld.states[gridworld.s2i(state)] == state

    @pytest.mark.parametrize("idx", [2, 14, 22])
    def test_i2s(self, gridworld, idx) -> None:
        assert gridworld.i2s(idx) == gridworld.states[idx]

    def test_bellman_operator(self, gridworld) -> None:
        A, b = gridworld.bellman_operator(0.9, lambda a, s: 0.25)
        assert b[gridworld.s2i((1, 2))] == 0
        assert b[gridworld.s2i((2, 4))] == -0.25
        assert b[gridworld.s2i((0, 3))] == 5
        assert A[gridworld.s2i((2, 4)), gridworld.s2i((2, 4))] == 0.9 * 0.25
        assert A[gridworld.s2i((1, 2)), gridworld.s2i((3, 3))] == 0
