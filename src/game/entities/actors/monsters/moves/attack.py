from typing import List

from game.effects.monster import MonsterEffect
from game.entities.actors.monsters.base import Intent
from game.entities.actors.monsters.base import Monster
from game.entities.actors.monsters.base import MonsterCollection
from game.entities.actors.monsters.moves.base import BaseMonsterMove
from game.entities.actors.characters.base import Character


class Attack(BaseMonsterMove):
    def __init__(self, value: int):
        self.value = value

    @property
    def intent(self) -> Intent:
        return Intent(damage=self.value)

    def __call__(
        self, owner: Monster, char: Character, monsters: MonsterCollection
    ) -> List[MonsterEffect]:
        return [MonsterEffect(owner, char, damage=self.value)]
