from abc import ABC
from typing import List, Optional

from game.battle.pipeline.pipeline import EffectPipeline
from game.battle.pipeline.steps.base import BaseStep
from game.effects.relic import RelicEffect
from game.entities.actors.characters.base import Character
from game.entities.actors.monsters.base import MonsterCollection


# TODO: does this need priority?
class BaseRelic(ABC):
    @property
    def step(self) -> Optional[BaseStep]:
        return None

    def on_turn_end(
        self, char: Character, monsters: MonsterCollection
    ) -> List[RelicEffect]:
        return []

    def on_turn_start(
        self, char: Character, monsters: MonsterCollection
    ) -> List[RelicEffect]:
        return []

    def on_battle_end(
        self, char: Character, monsters: MonsterCollection
    ) -> List[RelicEffect]:
        return []

    def on_battle_start(
        self, char: Character, monsters: MonsterCollection
    ) -> List[RelicEffect]:
        return []


class Relics(list):
    def __init__(self):
        super().__init__([])

    def add_relic(self, relic: BaseRelic, pipeline: EffectPipeline) -> None:
        if not isinstance(relic, BaseRelic):
            raise TypeError(
                f"Can only add objects of {BaseRelic} type. Received type: {type(relic)}"
            )

        # If the relic has a corresponding EffectPipeline step, add it
        if relic.step is not None:
            pipeline.add_step(relic.step)

        self.append(relic)
