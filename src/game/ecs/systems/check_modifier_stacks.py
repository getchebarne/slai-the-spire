from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import ActorHasModifiersComponent
from src.game.ecs.components.actors import ModifierMinimumStacksComponent
from src.game.ecs.components.actors import ModifierStacksComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: add max stacks


class CheckModifierStacks(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        for actor_entity_id, _ in manager.get_component(ActorComponent):
            actor_has_modifiers_component = manager.get_component_for_entity(
                actor_entity_id, ActorHasModifiersComponent
            )
            if actor_has_modifiers_component is not None:
                for modifier_entity_id in actor_has_modifiers_component.modifier_entity_ids:
                    modifier_stacks_component = manager.get_component_for_entity(
                        modifier_entity_id, ModifierStacksComponent
                    )
                    modifier_minimum_stakcs_component = manager.get_component_for_entity(
                        modifier_entity_id, ModifierMinimumStacksComponent
                    )
                    if (
                        modifier_stacks_component is not None
                        and modifier_minimum_stakcs_component is not None
                        and modifier_stacks_component.value
                        < modifier_minimum_stakcs_component.value
                    ):
                        # Destroy
                        manager.destroy_entity(modifier_entity_id)
                        actor_has_modifiers_component.modifier_entity_ids.remove(
                            modifier_entity_id
                        )
