from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class DispatchEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        dispatch_effect_entity_id = None
        for effect_entity_id, effect_to_be_dispatched_component in manager.get_component(
            EffectIsQueuedComponent
        ):
            # Dispatch highest priority effect
            if effect_to_be_dispatched_component.priority == 0:
                dispatch_effect_entity_id = effect_entity_id

            else:
                # Decrease priority. TODO: rename "priority" to something else
                effect_to_be_dispatched_component.priority -= 1

        if dispatch_effect_entity_id is not None:
            manager.remove_component(dispatch_effect_entity_id, EffectIsQueuedComponent)
            manager.add_component(dispatch_effect_entity_id, EffectIsDispatchedComponent())
