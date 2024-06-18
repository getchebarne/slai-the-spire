from src.game.ecs.components.actors import ModifierMinimumStacksComponent
from src.game.ecs.components.actors import ModifierParentComponent
from src.game.ecs.components.actors import ModifierStacksDurationComponent
from src.game.ecs.components.actors import ModifierWeakComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.components.effects import EffectCreateWeakComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: will probably have to abstract effect creation logic
class CreateModifierWeakSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(EffectIsTargetedComponent, EffectCreateWeakComponent)
        )

        if query_result:
            for target_entity_id, _ in manager.get_component(EffectTargetComponent):
                modifier_already_exists = False
                for _, (modifier_parent_component, _) in manager.get_components(
                    ModifierParentComponent, ModifierWeakComponent
                ):
                    if modifier_parent_component.actor_entity_id == target_entity_id:
                        # Modifier already exists for parent actor
                        modifier_already_exists = True
                        break

                if modifier_already_exists:
                    continue

                # Create modifier instance
                manager.create_entity(
                    NameComponent("Weak"),
                    ModifierParentComponent(target_entity_id),
                    ModifierWeakComponent(),
                    ModifierMinimumStacksComponent(1),
                    ModifierStacksDurationComponent(),
                )
