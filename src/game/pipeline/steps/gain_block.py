from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep
from game.context import entities


MAX_BLOCK = 999


class GainBlock(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        block = effect.value
        target_entity_id = effect.target_entity_id

        entities.loc[target_entity_id, "entity_current_block"] = min(
            MAX_BLOCK, entities.loc[target_entity_id, "entity_current_block"] + block
        )

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.BLOCK
