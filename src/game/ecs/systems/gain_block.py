from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus
from src.game.ecs.utils import resolve_effect_target_entities


class GainBlockSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            _,
            gain_block_effect_component,
            effect_query_components_component,
            effect_selection_type_component,
        ) = next(
            manager.get_components(
                EffectIsDispatchedComponent,
                GainBlockEffectComponent,
                EffectQueryComponentsComponent,
                EffectSelectionTypeComponent,
            )
        )
        # Resolve target entities
        target_entity_ids = resolve_effect_target_entities(
            effect_query_components_component.value, effect_selection_type_component.value, manager
        )
        block = gain_block_effect_component.value
        for target_entity_id in target_entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            block_component.current = min(block_component.current + block, block_component.max)

        return ProcessStatus.COMPLETE
