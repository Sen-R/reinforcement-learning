from typing import Iterable, Iterator, Tuple, Optional
from itertools import cycle
import logging
from copy import copy
import numpy as np
from dataclasses import dataclass
from .base import Environment


logger = logging.getLogger(__name__)


@dataclass
class BJState:
    """Dataclass representing the state of a Blackjack game."""

    current_count: int
    dealer_card: int
    usable_ace: bool

    def is_valid(self) -> bool:
        return (12 <= self.current_count <= 20) and (
            2 <= self.dealer_card <= 11
        )


class InfiniteDeck:
    """Class representing an infinite deck of cards."""

    _cards = np.arange(1, 11)
    _p = np.array([1] * 9 + [4]) / 13  # probability table for cards

    def __init__(self, random_state=None):
        self._rng = np.random.default_rng(random_state)

    def __iter__(self) -> "InfiniteDeck":
        return self

    def __next__(self) -> int:
        card = self._rng.choice(self._cards, p=self._p)
        logger.info("drawing card: %s", card)
        return card


class RepeatingDeck:
    """Returns an iterable representing a deck with the specified sequence,
    repeated indefinitely."""

    def __init__(self, contents: Iterable[int]):
        self._contents_gen = (card for card in cycle(contents))

    def __iter__(self) -> "RepeatingDeck":
        return self

    def __next__(self) -> int:
        card = next(self._contents_gen)
        logger.info("drawing card: %s", card)
        return card


class Blackjack(Environment):
    """Blackjack environment.

    Version of the game described in chapter 5 of Sutton, Barto (2018). Player
    plays against the dealer only. Dealer sticks for count of 17 or over.

    Card code: 1 represents an ace, 2--9 represent number cards and 10
    represents any face card.
    """

    HIT = 1
    STICK = 0

    def __init__(
        self, starting_state: BJState, deck: Optional[Iterator[int]] = None
    ):
        self._starting_state = copy(starting_state)
        self.deck = InfiniteDeck() if deck is None else deck
        self.reset()

    def act(self, action: int) -> int:
        if action == self.HIT and self._state.current_count <= 21:
            next_card = next(self.deck)
            (
                self._state.current_count,
                self._state.usable_ace,
            ) = Blackjack.update_count(
                self._state.current_count, self._state.usable_ace, next_card
            )
            if self._state.current_count > 21:
                return -1
            else:
                return 0
        elif action == self.STICK:
            dealer_usable_ace = self._state.dealer_card == 1
            while self._state.dealer_card < 17:
                next_card = next(self.deck)
                print(self._state, next_card)
                (
                    self._state.dealer_card,
                    dealer_usable_ace,
                ) = Blackjack.update_count(
                    self._state.dealer_card, dealer_usable_ace, next_card
                )
            print(self._state)
            if self._state.dealer_card > 21:
                return 1
            else:
                difference = (
                    self._state.current_count - self._state.dealer_card
                )
                if difference > 0:
                    return 1
                elif difference < 0:
                    return -1
                else:
                    return 0
        else:
            raise ValueError(f"Invalid action: {action}")

    def observe(self) -> BJState:
        return self._state

    def reset(self) -> None:
        # Nothing to reset
        self._state = copy(self._starting_state)

    @property
    def done(self) -> bool:
        return not self._state.is_valid()

    @property
    def state(self) -> BJState:
        return self._state

    @staticmethod
    def update_count(
        initial_count: int, usable_ace: bool, new_card: int
    ) -> Tuple[int, bool]:
        """Returns new card count, after adding `new_card` to the existing
        hand."""
        updated_count = initial_count + new_card
        if updated_count > 21 and usable_ace:
            updated_count -= 10
            usable_ace = False
        elif updated_count < 12 and new_card == 1:
            updated_count += 10
            usable_ace = True
        return updated_count, usable_ace
