from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectSetBlockToZero
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class SetBlockToZeroSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, _ = next(
                manager.get_components(EffectIsTargetedSingletonComponent, EffectSetBlockToZero)
            )

        except StopIteration:
            return

        # Get target entities
        target_entity_ids = [
            target_entity_id
            for target_entity_id, _ in manager.get_component(EffectTargetComponent)
        ]

        for target_entity_id in target_entity_ids:
            block_component = manager.get_component_for_entity(target_entity_id, BlockComponent)
            block_component.current = 0

        return
