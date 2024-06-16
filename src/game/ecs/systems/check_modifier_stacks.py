from src.game.ecs.components.creatures import CreatureComponent
from src.game.ecs.components.creatures import CreatureHasModifiersComponent
from src.game.ecs.components.creatures import ModifierMinimumStacksComponent
from src.game.ecs.components.creatures import ModifierStacksComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: add max stacks


class CheckModifierStacks(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        for creature_entity_id, _ in manager.get_component(CreatureComponent):
            creature_has_modifiers_component = manager.get_component_for_entity(
                creature_entity_id, CreatureHasModifiersComponent
            )
            if creature_has_modifiers_component is not None:
                for modifier_entity_id in creature_has_modifiers_component.modifier_entity_ids:
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
                        creature_has_modifiers_component.modifier_entity_ids.remove(
                            modifier_entity_id
                        )
