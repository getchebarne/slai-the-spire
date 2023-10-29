from game.effects.base import TargetType
from game.effects.card import CardEffect
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType


BASE_COST = 1
BASE_DAMAGE = 6


class Strike(BaseCard):
    name = "Strike"
    type_ = CardType.ATTACK
    effects = [CardEffect(target_type=TargetType.SINGLE, damage=BASE_DAMAGE)]

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)
