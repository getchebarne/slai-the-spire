from typing import List, Optional

from game.effects.card import CardEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType
from game.entities.cards.base import ensure_target_monster_idx


BASE_COST = 0
BASE_DAMAGE = 3
BASE_WEAK = 1


class Neutralize(BaseCard):
    name = "Neutralize"
    type_ = CardType.ATTACK

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)

    @ensure_target_monster_idx
    def use(
        self,
        char: Character,
        monsters: MonsterGroup,
        target_monster_idx: Optional[int] = None,
    ) -> List[CardEffect]:
        target_monster = monsters[target_monster_idx]
        return [CardEffect(char, target_monster, damage=BASE_DAMAGE, weak=BASE_WEAK)]
