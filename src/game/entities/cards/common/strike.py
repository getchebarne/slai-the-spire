from typing import List, Optional

from game.effects.card import CardEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.base import MonsterCollection
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType
from game.entities.cards.base import ensure_target_monster_idx


BASE_COST = 1
BASE_DAMAGE = 6


class Strike(BaseCard):
    name = "Strike"
    type_ = CardType.ATTACK

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)

    @ensure_target_monster_idx
    def use(
        self,
        char: Character,
        monsters: MonsterCollection,
        target_monster_idx: Optional[int] = None,
    ) -> List[CardEffect]:
        target_monster = monsters[target_monster_idx]
        return [CardEffect(char, target_monster, damage=BASE_DAMAGE)]
