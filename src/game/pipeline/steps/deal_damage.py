from game import context
from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep


class DealDamage(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        damage = effect.value
        target_entity_id = effect.target_entity_id

        # Remove block
        dmg_over_block = max(0, damage - context.entities[target_entity_id].current_block)
        context.entities[target_entity_id].current_block = max(
            0, context.entities[target_entity_id].current_block - damage
        )
        # Remove health
        context.entities[target_entity_id].current_health = max(
            0, context.entities[target_entity_id].current_health - dmg_over_block
        )

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.DAMAGE
