from abc import ABC
from typing import List, Optional

from game.effects.relic import RelicEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.group import MonsterGroup
from game.pipeline.steps.base import BaseStep


# TODO: does this need priority?
class BaseRelic(ABC):
    @property
    def step(self) -> Optional[BaseStep]:
        return None

    # TODO: maybe rename to `on_char_turn_end`
    def on_turn_end(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        return []

    # TODO: maybe rename to `on_char_turn_start`
    def on_turn_start(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        return []

    def on_battle_end(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        return []

    def on_battle_start(self, char: Character, monsters: MonsterGroup) -> List[RelicEffect]:
        return []
