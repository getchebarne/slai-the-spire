import random

from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectToBeTargetedComponent
from src.game.ecs.components.target import TargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


# TODO: add effect source to trigger additional effects (e.g., thorns)
# TODO: make sure effects are triggered in the correct order
class TargetEffectsSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        for effect_entity_id, (
            effect_to_be_targeted_component,
            effect_selection_type_component,
            effect_query_components_component,
        ) in manager.get_components(
            EffectToBeTargetedComponent,
            EffectSelectionTypeComponent,
            EffectQueryComponentsComponent,
        ):
            # Only dispatch highest priority effect
            if effect_to_be_targeted_component.priority != 0:
                # TODO: maybe this is wrong, the effect may not be completely processed
                effect_to_be_targeted_component.priority -= 1
                continue

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

            elif effect_selection_type_component.value == EffectSelectionType.SPECIFIC:
                # TODO: there can be multiple specific targets (e.g., prep+)
                query_result = list(manager.get_component(TargetComponent))
                if len(query_result) > 1:
                    raise ValueError("Too many entities to apply effect")

                target_entity_id, _ = query_result[0]

                # Verify that the target entity is in the query target entities
                if target_entity_id not in query_target_entity_ids:
                    raise ValueError(
                        f"Target entity {target_entity_id} not in query target entities"
                    )

                target_entity_ids = [target_entity_id]

            elif effect_selection_type_component.value == EffectSelectionType.RANDOM:
                target_entity_ids = [random.choice(query_target_entity_ids)]

            elif effect_selection_type_component.value == EffectSelectionType.ALL:
                target_entity_ids = query_target_entity_ids

            # Duplicate the effect entity so that it can be manipulated by the rest of the
            # pipeline and add a component with its target entities
            manager.add_component(
                manager.duplicate_entity(effect_entity_id),
                EffectApplyToComponent(entity_ids=target_entity_ids),
            )
            # Untag the effect
            manager.remove_component(effect_entity_id, EffectToBeTargetedComponent)

        return ProcessStatus.COMPLETE
