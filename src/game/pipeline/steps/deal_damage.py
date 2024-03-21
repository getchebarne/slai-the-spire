from game.core.effect import Effect
from game.core.effect import EffectType
from game.pipeline.steps.base import BaseStep
from game.context import entities


class DealDamage(BaseStep):
    def _apply_effect(self, effect: Effect) -> None:
        damage = effect.value
        target_entity_id = effect.target_entity_id

        # Remove block
        dmg_over_block = max(0, damage - entities.loc[target_entity_id, "entity_current_block"])
        entities.loc[target_entity_id, "entity_current_block"] = max(
            0, entities.loc[target_entity_id, "entity_current_block"] - damage
        )
        # Remove health
        entities.loc[target_entity_id, "entity_current_health"] = max(
            0, entities.loc[target_entity_id, "entity_current_health"] - dmg_over_block
        )

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.DAMAGE
