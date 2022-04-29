import pytest
from rl.environments.base import Environment
from rl.environments.blackjack import BlackJack, BJState


class TestBlackJackState:
    def test_fields(self) -> None:
        state = BJState(current_count=12, dealer_card=0, usable_ace=False)
        assert state.current_count == 12
        assert state.dealer_card == 0
        assert usable_ace == False

    @pytest.mark.parametrize(
        "state,is_valid",
        [
            [BJState(12, 0, False), True],
            [BJState(11, 3, False), False],
            [BJState(21, 0, False), True],
            [BJState(22, 0, False), False],
            [BJState(15, -1, False), False],
            [BJState(15, 10, False), True],
            [BJState(15, 11, False), False],
            [BJState(12, 9, True), True],
        ],
    )
    def test_is_valid_method(self, state: BJState, is_valid: bool) -> None:
        assert state.is_valid() == is_valid

class TestBlackJack:
    def test_blackjack_is_instance_of_environment(self) -> None:
        assert isinstance(BlackJack(), Environment)

    def test_blackjack_can_set_random_state(self) -> None:
        BlackJack(random_state=42)

    def test_blackjack_set_starting_state(self) -> None:
        starting_state = BJState(15, 10, False)
        bj = BlackJack(starting_state=starting_state)
        assert bj.observe() == starting_state

    def test_blackjack_action_codes(self) -> None:
        assert BlackJack.HIT == 1
        assert BlackJack.STICK == 0

    def test_blackjack_act_method(self) -> None:
        bj = BlackJack(random_state=42)
        orig_state = bj.observe()
        bj.act(BlackJack.HIT)
        new_state = bj.observe()
        assert orig_state.current_count < new_state.current_count
        assert orig_state.dealer_card == new_state.dealer_card

    def test_state_signal_has_right_form(self) -> None:
        state = BlackJack(random_state=42).observe()
        assert isinstance(state, BJState)
        assert state.is_valid()

    def test_reset_method(self) -> None:
        # Should reset to initial random state
        bj1 = BlackJack(random_state=42)
        state_0 = bj1.observe()
        bj1.reset()
        state_1 = bj1.observe()
        assert state_0 == state_1

    def test_done_is_false_at_beginning(self) -> None:
        bj = BlackJack(random_state=42)
        assert not bj.done()

