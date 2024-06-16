from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.creatures import CreatureHasModifiersComponent
from src.game.ecs.components.creatures import ModifierMinimumStacksComponent
from src.game.ecs.components.creatures import ModifierStacksDurationComponent
from src.game.ecs.components.creatures import ModifierWeakComponent
from src.game.ecs.components.effects import EffectCreateWeakComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class CreateModifierWeakSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(EffectIsTargetedComponent, EffectCreateWeakComponent)
        )

        if query_result:
            for target_entity_id, _ in manager.get_component(EffectTargetComponent):
                creature_has_modifiers_component = manager.get_component_for_entity(
                    target_entity_id, CreatureHasModifiersComponent
                )
                if creature_has_modifiers_component is None:
                    creature_has_modifiers_component = CreatureHasModifiersComponent([])
                    manager.add_component(target_entity_id, creature_has_modifiers_component)

                for modifier_entity_id in creature_has_modifiers_component.modifier_entity_ids:
                    if (
                        manager.get_component_for_entity(modifier_entity_id, ModifierWeakComponent)
                        is not None
                    ):
                        return

                # Create a weak modifier entity
                creature_has_modifiers_component.modifier_entity_ids.append(
                    manager.create_entity(
                        NameComponent("Weak"),
                        ModifierWeakComponent(),
                        ModifierMinimumStacksComponent(1),
                        ModifierStacksDurationComponent(),
                    )
                )
