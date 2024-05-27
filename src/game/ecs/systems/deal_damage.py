from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.effects import DealDamageEffectComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus
from src.game.ecs.utils import resolve_effect_target_entities


class DealDamageSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            _,
            deal_damage_effect_component,
            effect_query_components_component,
            effect_selection_type_component,
        ) = next(
            manager.get_components(
                EffectIsDispatchedComponent,
                DealDamageEffectComponent,
                EffectQueryComponentsComponent,
                EffectSelectionTypeComponent,
            )
        )

        # Resolve target entities
        target_entity_ids = resolve_effect_target_entities(
            effect_query_components_component.value, effect_selection_type_component.value, manager
        )

        damage = deal_damage_effect_component.value
        for target_entity_id in target_entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            health_component = manager.get_component_for_entity(target_entity_id, HealthComponent)

            # Remove block
            dmg_over_block = max(0, damage - block_component.current)
            block_component.current = max(0, block_component.current - damage)

            # Remove health
            health_component.current = max(0, health_component.current - dmg_over_block)

        return ProcessStatus.COMPLETE
