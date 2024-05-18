from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.ecs.components import BlockComponent
from src.game.ecs.components import HealthComponent
from src.game.ecs.manager import ECSManager
from src.game.pipeline.steps.base import BaseStep


class DealDamage(BaseStep):
    def _apply_effect(self, manager: ECSManager, target_entity_id: int, effect: Effect) -> None:
        damage = effect.value
        block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
        health_component = manager.get_component_for_entity(target_entity_id, HealthComponent)

        # Remove block
        dmg_over_block = max(0, damage - block_component.current)
        block_component.current = max(0, block_component.current - damage)

        # Remove health
        health_component.current = max(0, health_component.current - dmg_over_block)

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.DAMAGE
