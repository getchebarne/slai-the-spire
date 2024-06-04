from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: think about changing how the whole effect duplication thing is implemented
class DestroyEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, _ = next(manager.get_component(EffectIsTargetedComponent))
            manager.destroy_entity(effect_entity_id)

        except StopIteration:
            return

        return
