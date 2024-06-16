from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.creatures import CreatureHasModifiersComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: change name
class TagCardTargetModifiersSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            _, (_, creature_has_modifiers_component) = next(
                manager.get_components(CardTargetComponent, CreatureHasModifiersComponent)
            )

        except StopIteration:
            return

        # TODO: maybe move this
        for modifier_entity_id in creature_has_modifiers_component.modifier_entity_ids:
            manager.add_component(modifier_entity_id, CardTargetComponent())
