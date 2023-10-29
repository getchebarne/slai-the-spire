from game.effects.base import TargetType
from game.effects.card import CardEffect
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType


BASE_COST = 0
BASE_DAMAGE = 3
BASE_WEAK = 1


class Neutralize(BaseCard):
    name = "Neutralize"
    type_ = CardType.ATTACK
    effects = [
        CardEffect(target_type=TargetType.SINGLE, damage=BASE_DAMAGE, weak=BASE_WEAK)
    ]

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)
