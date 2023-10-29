from typing import List

from game.entities.cards.base import BaseCard
from game.entities.cards.common.strike import Strike
from game.entities.cards.common.defend import Defend
from game.entities.cards.silent.neutralize import Neutralize


class Deck:
    def __init__(self, cards: List[BaseCard]):
        self.cards = cards


SILENT_STARTER_DECK = Deck([Strike()] * 6 + [Defend()] * 6 + [Neutralize()])
