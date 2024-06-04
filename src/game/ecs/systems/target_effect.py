import random

from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectIsWaitingInputTargetComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: check targeted entities are alive
# TODO: query dispatched effects only, check if they need to be targeted here
class TargetEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, _ = next(manager.get_component(EffectIsDispatchedComponent))

        except StopIteration:
            return

        effect_query_components_component = manager.get_component_for_entity(
            effect_entity_id, EffectQueryComponentsComponent
        )
        effect_selection_type_component = manager.get_component_for_entity(
            effect_entity_id, EffectSelectionTypeComponent
        )
        # Remove previous tags TODO: revist
        manager.destroy_component(EffectTargetComponent)

        # TODO: improve this, it's horrendous
        if (
            effect_query_components_component is not None
            and effect_selection_type_component is not None
        ):
            # Get the entities that match the effect's potential targets
            query_target_entity_ids = [
                query_entity_id
                for query_entity_id, _ in manager.get_components(
                    *effect_query_components_component.value
                )
            ]
            # Resolve target entity
            if effect_selection_type_component.value == EffectSelectionType.NONE:
                if len(query_target_entity_ids) > 1:
                    raise ValueError("Too many entities to apply effect")

                target_entity_ids = query_target_entity_ids

            elif effect_selection_type_component.value == EffectSelectionType.ALL:
                target_entity_ids = query_target_entity_ids

            elif effect_selection_type_component.value == EffectSelectionType.RANDOM:
                target_entity_ids = [random.choice(query_target_entity_ids)]

            elif effect_selection_type_component.value == EffectSelectionType.SPECIFIC:
                if len(query_target_entity_ids) <= 1:
                    target_entity_ids = query_target_entity_ids

                else:
                    # TODO: fix, this is broken
                    is_selected_entity_ids = [
                        is_selected_entity_id
                        for is_selected_entity_id, _ in manager.get_component(IsSelectedComponent)
                    ]
                    if set(is_selected_entity_ids).issubset(set(query_target_entity_ids)):
                        target_entity_ids = is_selected_entity_ids

                    else:
                        manager.add_component(
                            effect_entity_id, EffectIsWaitingInputTargetComponent()
                        )
                        return

            # Tag target entities
            for target_entity_id in target_entity_ids:
                manager.add_component(target_entity_id, EffectTargetComponent())

        # Untag & retag
        manager.remove_component(effect_entity_id, EffectIsDispatchedComponent)
        manager.add_component(effect_entity_id, EffectIsTargetedComponent())
