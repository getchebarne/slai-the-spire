import random

from src.game.ecs.components.base import BaseComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.target import TargetComponent
from src.game.ecs.manager import ECSManager


def resolve_effect_target_entities(
    effect_query_components: list[BaseComponent],
    effect_selection_type: EffectSelectionType,
    manager: ECSManager,
) -> list[int]:
    # Get the entities that match the effect's potential targets
    query_target_entity_ids = [
        query_entity_id for query_entity_id, _ in manager.get_components(*effect_query_components)
    ]
    # Resolve target entity
    if effect_selection_type == EffectSelectionType.NONE:
        if len(query_target_entity_ids) > 1:
            raise ValueError("Too many entities to apply effect")

        return query_target_entity_ids

    if effect_selection_type == EffectSelectionType.SPECIFIC:
        # TODO: there can be multiple specific targets (e.g., prep+)
        target_entity_ids = []
        for target_entity_id, _ in manager.get_component(TargetComponent):
            if target_entity_id not in query_target_entity_ids:
                raise ValueError(f"Target entity {target_entity_id} not in query target entities")

            target_entity_ids.append(target_entity_id)

        return target_entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(query_target_entity_ids)]

    if effect_selection_type == EffectSelectionType.ALL:
        return query_target_entity_ids
