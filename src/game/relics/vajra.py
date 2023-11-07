from typing import List

from game.effects.relic import RelicEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.base import MonsterCollection
from game.relics.base import BaseRelic


PLUS_STR = 1


class Vajra(BaseRelic):
    def on_battle_start(
        self, char: Character, monsters: MonsterCollection
    ) -> List[RelicEffect]:
        return [RelicEffect(char, char, plus_str=PLUS_STR)]
