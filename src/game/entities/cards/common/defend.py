from game.effects.base import TargetType
from game.effects.card import CardEffect
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType


BASE_COST = 1
BASE_BLOCK = 5


class Defend(BaseCard):
    name = "Defend"
    type_ = CardType.SKILL
    effects = [CardEffect(target_type=TargetType.SELF, block=BASE_BLOCK)]

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)
