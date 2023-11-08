from typing import List, Optional

from game.effects.card import CardEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.cards.base import BaseCard
from game.entities.cards.base import CardType


BASE_COST = 1
BASE_BLOCK = 5


class Defend(BaseCard):
    name = "Defend"
    type_ = CardType.SKILL

    def __init__(self, cost: int = BASE_COST):
        super().__init__(cost)

    def use(
        self,
        char: Character,
        monsters: MonsterGroup,
        target_monster_idx: Optional[int] = None,
    ) -> List[CardEffect]:
        return [CardEffect(char, char, block=BASE_BLOCK)]
