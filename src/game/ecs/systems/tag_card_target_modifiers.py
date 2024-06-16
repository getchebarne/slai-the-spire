from src.game.ecs.components.actors import ActorHasModifiersComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: change name
class TagCardTargetModifiersSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            _, (_, actor_has_modifiers_component) = next(
                manager.get_components(CardTargetComponent, ActorHasModifiersComponent)
            )

        except StopIteration:
            return

        # TODO: maybe move this
        for modifier_entity_id in actor_has_modifiers_component.modifier_entity_ids:
            manager.add_component(modifier_entity_id, CardTargetComponent())
