from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import ModifierParentActorComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: change name
class TagCardTargetModifiersSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            actor_entity_id, _ = next(manager.get_components(ActorComponent, CardTargetComponent))

        except StopIteration:
            return

        # TODO: maybe move this
        for modifier_entity_id, modifier_parent_actor_component in manager.get_component(
            ModifierParentActorComponent
        ):
            if modifier_parent_actor_component.actor_entity_id == actor_entity_id:
                manager.add_component(modifier_entity_id, CardTargetComponent())
