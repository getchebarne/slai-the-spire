from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class GainBlockSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        effect_entity_ids_to_untag = []
        for effect_entity_id, (
            gain_block_effect_component,
            effect_apply_to_component,
        ) in manager.get_components(GainBlockEffectComponent, EffectApplyToComponent):
            for target_entity_id in effect_apply_to_component.entity_ids:
                manager.get_component_for_entity(
                    target_entity_id, BlockComponent
                ).current += gain_block_effect_component.value

            effect_entity_ids_to_untag.append(effect_entity_id)

        for effect_entity_id in effect_entity_ids_to_untag:
            manager.remove_component(effect_entity_id, EffectApplyToComponent)


class DealDamageSystem(BaseSystem):
    def __call__(self, manager: ECSManager) -> None:
        effect_entity_ids_to_untag = []
        for effect_entity_id, (
            deal_damage_effect_component,
            effect_apply_to_component,
        ) in manager.get_components(DealDamageEffectComponent, EffectApplyToComponent):
            for target_entity_id in effect_apply_to_component.entity_ids:
                damage = deal_damage_effect_component.value
                block_component = manager.get_component_for_entity(
                    target_entity_id, BlockComponent
                )
                health_component = manager.get_component_for_entity(
                    target_entity_id, HealthComponent
                )
                # Remove block
                dmg_over_block = max(0, damage - block_component.current)
                block_component.current = max(0, block_component.current - damage)

                # Remove health
                health_component.current = max(0, health_component.current - dmg_over_block)

            effect_entity_ids_to_untag.append(effect_entity_id)

        for effect_entity_id in effect_entity_ids_to_untag:
            manager.remove_component(effect_entity_id, EffectApplyToComponent)
