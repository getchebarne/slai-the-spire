from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


# TODO: revise if effects need to be dispatched every loop?
class DispatchEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        for effect_entity_id, effect_to_be_dispatched_component in manager.get_component(
            EffectToBeDispatchedComponent
        ):
            # Only dispatch highest priority effect
            if effect_to_be_dispatched_component.priority == 0:
                manager.add_component(effect_entity_id, EffectIsDispatchedComponent())
                untag_effect_entity_id = effect_entity_id

            else:
                # TODO: maybe this is wrong, the effect may not be completely processed
                effect_to_be_dispatched_component.priority -= 1

        # Untag the effect that was dispatched
        manager.remove_component(untag_effect_entity_id, EffectToBeDispatchedComponent)

        return ProcessStatus.COMPLETE