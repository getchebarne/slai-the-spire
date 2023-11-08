from typing import List

from game.effects.monster import MonsterEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.moves.base import Intent
from game.entities.actors.monsters.base import Monster
from game.entities.actors.monsters.group import MonsterGroup
from game.entities.actors.monsters.moves.base import BaseMonsterMove


class Defend(BaseMonsterMove):
    def __init__(self, value: int):
        self.value = value

    @property
    def intent(self) -> Intent:
        return Intent(block=True)

    def __call__(
        self, owner: Monster, char: Character, monsters: MonsterGroup
    ) -> List[MonsterEffect]:
        return [MonsterEffect(owner, owner, block=self.value)]
