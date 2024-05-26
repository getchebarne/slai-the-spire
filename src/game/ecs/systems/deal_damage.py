from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class DealDamageSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            deal_damage_effect_component,
            effect_apply_to_component,
        ) = next(manager.get_components(DealDamageEffectComponent, EffectApplyToComponent))

        damage = deal_damage_effect_component.value
        for target_entity_id in effect_apply_to_component.entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            health_component = manager.get_component_for_entity(target_entity_id, HealthComponent)

            # Remove block
            dmg_over_block = max(0, damage - block_component.current)
            block_component.current = max(0, block_component.current - damage)

            # Remove health
            health_component.current = max(0, health_component.current - dmg_over_block)

        return ProcessStatus.COMPLETE
