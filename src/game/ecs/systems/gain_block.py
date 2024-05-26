from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class GainBlockSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            gain_block_effect_component,
            effect_apply_to_component,
        ) = next(manager.get_components(GainBlockEffectComponent, EffectApplyToComponent))

        block = gain_block_effect_component.value
        for target_entity_id in effect_apply_to_component.entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)

            block_component.current = min(block_component.current + block, block_component.max)

        return ProcessStatus.COMPLETE
