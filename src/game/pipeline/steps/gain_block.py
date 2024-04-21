from game.context import Context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep


MAX_BLOCK = 999


class GainBlock(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        block = effect.value
        target_entity_id = effect.target_entity_id

        context.entities[target_entity_id].current_block = min(
            MAX_BLOCK, context.entities[target_entity_id].current_block + block
        )

    def _condition(self, context: Context, effect: Effect) -> bool:
        return effect.type == EffectType.BLOCK
