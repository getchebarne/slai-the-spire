from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: revisit if effects need to be dispatched every loop?
class DispatchEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        dispatch_effect_entity_id = None
        for effect_entity_id, effect_to_be_dispatched_component in manager.get_component(
            EffectToBeDispatchedComponent
        ):
            # Only dispatch highest priority effect
            if effect_to_be_dispatched_component.priority == 0:
                dispatch_effect_entity_id = effect_entity_id

            else:
                # TODO: maybe this is wrong, the effect may not be completely processed
                effect_to_be_dispatched_component.priority -= 1

        if dispatch_effect_entity_id is not None:
            print(manager.get_entity(dispatch_effect_entity_id))
            # Untag the dispatched effect
            manager.remove_component(dispatch_effect_entity_id, EffectToBeDispatchedComponent)

            # Duplicate dispatched effect so that it can be modified by the downstream systems and
            # tag it
            manager.add_component(
                manager.duplicate_entity(dispatch_effect_entity_id), EffectIsDispatchedComponent()
            )
