from typing import Tuple, List
import pytest
from rl.environments.base import Environment
from rl.environments.blackjack import (
    Blackjack,
    BJState,
    InfiniteDeck,
    RepeatingDeck,
)


class TestBlackjackState:
    def test_fields(self) -> None:
        state = BJState(current_count=12, dealer_card=0, usable_ace=False)
        assert state.current_count == 12
        assert state.dealer_card == 0
        assert state.usable_ace is False

    @pytest.mark.parametrize(
        "state,is_valid",
        [
            [BJState(12, 0, False), False],
            [BJState(12, 1, False), False],
            [BJState(12, 2, False), True],
            [BJState(11, 3, False), False],
            [BJState(21, 3, False), False],
            [BJState(22, 3, False), False],
            [BJState(15, 12, False), False],
            [BJState(15, 10, False), True],
            [BJState(15, 11, False), True],
            [BJState(12, 9, True), True],
        ],
    )
    def test_is_valid_method(self, state: BJState, is_valid: bool) -> None:
        assert state.is_valid() == is_valid


class TestInfiniteDeck:
    def test_next_card(self) -> None:
        deck = InfiniteDeck(random_state=42)
        card = next(deck)
        assert 1 <= card <= 10


class TestRepeatingDeck:
    def test_next_card(self) -> None:
        deck = RepeatingDeck([1, 9])
        cards = [card for card, _ in zip(deck, range(5))]  # draw 5 cards
        assert cards == [1, 9, 1, 9, 1]


class TestUpdateCount:
    @pytest.mark.parametrize(
        "initial,new_card,final",
        [
            [(12, False), 3, (15, False)],
            [(19, False), 3, (22, False)],
            [(19, True), 3, (12, False)],
            [(12, False), 1, (13, False)],
            [(20, False), 1, (21, False)],
            [(12, True), 1, (13, True)],
            [(20, True), 1, (21, True)],
            [(3, False), 1, (14, True)],
            [(10, False), 1, (21, True)],
        ],
    )
    def test_update_card_count(
        self, initial: Tuple[int, bool], new_card: int, final: Tuple[int, bool]
    ) -> None:
        assert (
            Blackjack.update_count(initial[0], initial[1], new_card) == final
        )


class TestBlackjack:
    def test_is_instance_of_environment(self) -> None:
        starting_state = BJState(12, 10, False)  # arbitrary state
        assert isinstance(Blackjack(starting_state), Environment)

    def test_set_custom_deck(self) -> None:
        starting_state = BJState(12, 10, False)
        bj = Blackjack(starting_state, deck=RepeatingDeck([0, 9]))
        assert next(bj.deck) == 0
        assert next(bj.deck) == 9

    def test_set_starting_state(self) -> None:
        starting_state = BJState(15, 10, False)
        bj = Blackjack(starting_state=starting_state)
        assert bj.observe() == starting_state

    def test_default_initialisation_deals_two_cards_from_deck(self) -> None:
        deck = RepeatingDeck([1, 5, 7, 2])
        bj = Blackjack(deck=deck)
        assert bj.state == BJState(16, 7, True)

    def test_action_codes(self) -> None:
        assert Blackjack.HIT == 1
        assert Blackjack.STICK == 0

    def test_observe_method(self) -> None:
        bj = Blackjack(BJState(12, 9, False))
        assert bj.state == BJState(12, 9, False)

    @pytest.mark.parametrize(
        "orig_state,action,deck,fin_state,reward",
        [
            [BJState(14, 8, False), 1, [5], BJState(19, 8, False), 0],
            [BJState(17, 8, False), 0, [5, 7], BJState(17, 20, False), -1],
            [BJState(20, 8, False), 0, [5, 6], BJState(20, 19, False), 1],
            [BJState(21, 10, False), 0, [1], BJState(21, 21, False), 0],
            [BJState(19, 9, False), 1, [5], BJState(24, 9, False), -1],
            [BJState(19, 10, False), 0, [6, 9], BJState(19, 25, False), 1],
        ],
    )
    def test_act_method(
        self,
        orig_state: BJState,
        action: int,
        deck: List[int],
        fin_state: BJState,
        reward: int,
    ) -> None:
        bj = Blackjack(orig_state, deck=RepeatingDeck(deck))
        assert bj.observe() == orig_state
        actual_reward = bj.act(action)
        assert bj.observe() == fin_state
        assert actual_reward == reward

    @pytest.mark.parametrize(
        "state,done",
        [
            [BJState(14, 8, False), False],
            [BJState(22, 8, False), True],
            [BJState(20, 17, False), True],
            [BJState(21, 7, False), True],
            [BJState(19, 25, False), True],
        ],
    )
    def test_done_method(self, state: BJState, done: bool) -> None:
        bj = Blackjack(state)
        assert bj.done is done

    def test_reset_method(self) -> None:
        starting_state = BJState(14, 9, False)
        bj = Blackjack(starting_state, deck=RepeatingDeck([2]))
        bj.act(Blackjack.HIT)
        assert bj.state != starting_state
        bj.reset()
        assert bj.state == starting_state
