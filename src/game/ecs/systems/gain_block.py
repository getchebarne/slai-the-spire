from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.effects import EffectGainBlockComponent
from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class GainBlockSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, (_, gain_block_effect_component) = next(
                manager.get_components(
                    EffectIsTargetedSingletonComponent, EffectGainBlockComponent
                )
            )

        except StopIteration:
            return

        block = gain_block_effect_component.value
        for target_entity_id, _ in manager.get_component(EffectTargetComponent):
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            block_component.current = min(block_component.current + block, block_component.max)

        return
