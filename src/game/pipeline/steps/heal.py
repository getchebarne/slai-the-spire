from src.game.context import Context
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.pipeline.steps.base import BaseStep


class Heal(BaseStep):
    def _apply_effect(self, context: Context, effect: Effect) -> None:
        heal = effect.value
        target_entity_id = effect.target_entity_id

        context.entities[target_entity_id].current_health = min(
            context.entities[target_entity_id].current_health + heal,
            context.entities[target_entity_id].max_health,
        )

    def _condition(self, context: Context, effect: Effect) -> bool:
        return effect.type == EffectType.HEAL
