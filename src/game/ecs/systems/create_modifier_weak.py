from src.game.ecs.components.actors import ActorHasModifiersComponent
from src.game.ecs.components.actors import ModifierMinimumStacksComponent
from src.game.ecs.components.actors import ModifierStacksDurationComponent
from src.game.ecs.components.actors import ModifierWeakComponent
from src.game.ecs.components.common import NameComponent
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
                actor_has_modifiers_component = manager.get_component_for_entity(
                    target_entity_id, ActorHasModifiersComponent
                )
                if actor_has_modifiers_component is None:
                    actor_has_modifiers_component = ActorHasModifiersComponent([])
                    manager.add_component(target_entity_id, actor_has_modifiers_component)

                for modifier_entity_id in actor_has_modifiers_component.modifier_entity_ids:
                    if (
                        manager.get_component_for_entity(modifier_entity_id, ModifierWeakComponent)
                        is not None
                    ):
                        return

                # Create a weak modifier entity
                actor_has_modifiers_component.modifier_entity_ids.append(
                    manager.create_entity(
                        NameComponent("Weak"),
                        ModifierWeakComponent(),
                        ModifierMinimumStacksComponent(1),
                        ModifierStacksDurationComponent(),
                    )
                )
