from src.game.ecs.components.creatures import BlockComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.components.effects import GainBlockEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class GainBlockSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        try:
            effect_entity_id, (_, gain_block_effect_component) = next(
                manager.get_components(EffectIsTargetedComponent, GainBlockEffectComponent)
            )

        except StopIteration:
            return ProcessStatus.PASS

        # Get target entities
        target_entity_ids = [
            target_entity_id
            for target_entity_id, _ in manager.get_component(EffectTargetComponent)
        ]
        block = gain_block_effect_component.value
        for target_entity_id in target_entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            block_component.current = min(block_component.current + block, block_component.max)

        return ProcessStatus.COMPLETE
