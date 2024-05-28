import random

from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class TargetEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            _,
            effect_query_components_component,
            effect_selection_type_component,
        ) = next(
            manager.get_components(
                EffectIsDispatchedComponent,
                EffectQueryComponentsComponent,
                EffectSelectionTypeComponent,
            )
        )

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

        if effect_selection_type_component.value == EffectSelectionType.ALL:
            target_entity_ids = query_target_entity_ids

        if effect_selection_type_component.value == EffectSelectionType.RANDOM:
            target_entity_ids = [random.choice(query_target_entity_ids)]

        # Tag target entities
        for target_entity_id in target_entity_ids:
            manager.add_component(target_entity_id, EffectTargetComponent)
