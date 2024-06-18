from src.game.ecs.components.actors import ModifierMinimumStacksComponent
from src.game.ecs.components.actors import ModifierStacksComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ApplyModifierDeltaSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(EffectIsTargetedComponent, EffectModifierDeltaComponent)
        )

        if query_result:
            _, (_, effect_modifer_delta_component) = query_result[0]

            destroy_entity_ids = []
            for target_entity_id, _ in manager.get_component(EffectTargetComponent):
                modifier_stacks_component = manager.get_component_for_entity(
                    target_entity_id, ModifierStacksComponent
                )
                if modifier_stacks_component is None:
                    modifier_stacks_component = ModifierStacksComponent(
                        effect_modifer_delta_component.value
                    )
                    manager.add_component(target_entity_id, modifier_stacks_component)

                else:
                    modifier_stacks_component.value += effect_modifer_delta_component.value

                modifier_minimum_stacks_component = manager.get_component_for_entity(
                    target_entity_id, ModifierMinimumStacksComponent
                )
                if (
                    modifier_minimum_stacks_component is not None
                    and modifier_stacks_component.value < modifier_minimum_stacks_component.value
                ):
                    destroy_entity_ids.append(target_entity_id)

            for destroy_entity_id in destroy_entity_ids:
                manager.destroy_entity(destroy_entity_id)
