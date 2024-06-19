import random
from typing import Optional

from src.game.ecs.components.effects import EffectInputTargetComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectIsHaltedComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectNumberOfTargetsComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


def _handle_effect_selection_component_none(query_target_entity_ids: list[int]) -> list[int]:
    if len(query_target_entity_ids) > 1:
        raise ValueError("Too many entities to apply effect")

    return query_target_entity_ids


def _handle_effect_selection_type_all(query_target_entity_ids: list[int]) -> list[int]:
    return query_target_entity_ids


def _handle_effect_selection_type_random(query_target_entity_ids: list[int]) -> list[int]:
    return [random.choice(query_target_entity_ids)]


def _handle_effect_selection_type_specific(
    query_target_entity_ids: list[int], effect_entity_id: int, manager: ECSManager
) -> Optional[list[int]]:
    if len(query_target_entity_ids) <= 1:
        return query_target_entity_ids

    # Get number of targets
    num_targets = manager.get_component_for_entity(
        effect_entity_id, EffectNumberOfTargetsComponent
    ).value

    # Get currently selected entity ids
    is_selected_entity_ids = [
        is_selected_entity_id
        for is_selected_entity_id, _ in manager.get_component(EffectInputTargetComponent)
    ]

    # Check if the effect's target entities have been selected
    if len(is_selected_entity_ids) == num_targets and all(
        [
            is_selected_entity_id in query_target_entity_ids
            for is_selected_entity_id in is_selected_entity_ids
        ]
    ):
        # Re-queue halted effects
        for is_halted_effect_entity_id, effect_is_halted_component in manager.get_component(
            EffectIsHaltedComponent
        ):
            manager.add_component(
                is_halted_effect_entity_id,
                EffectIsQueuedComponent(effect_is_halted_component.position),
            )

        manager.destroy_component(EffectIsHaltedComponent)
        manager.destroy_component(EffectIsPendingInputTargetsComponent)

        return is_selected_entity_ids

    # Tag effect as pending input targets
    manager.add_component(effect_entity_id, EffectIsPendingInputTargetsComponent())

    # Halt queued effects
    for is_queued_effect_entity_id, effect_is_queued_component in manager.get_component(
        EffectIsQueuedComponent
    ):
        manager.add_component(
            is_queued_effect_entity_id,
            EffectIsHaltedComponent(effect_is_queued_component.position),
        )

    manager.destroy_component(EffectIsQueuedComponent)


# TODO: check targeted entities are alive
class TargetEffectSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, _ = next(manager.get_component(EffectIsDispatchedComponent))

        except StopIteration:
            return

        # Destroy previous effect's target tags
        manager.destroy_component(EffectTargetComponent)

        # Get effect's targeting data (if it exists)
        effect_query_components_component = manager.get_component_for_entity(
            effect_entity_id, EffectQueryComponentsComponent
        )
        effect_selection_type_component = manager.get_component_for_entity(
            effect_entity_id, EffectSelectionTypeComponent
        )

        # TODO: improve this, it's horrendous
        if effect_query_components_component is not None:
            # Get the entities that match the effect's potential target entities
            query_target_entity_ids = [
                query_entity_id
                for query_entity_id, _ in manager.get_components(
                    *effect_query_components_component.value
                )
            ]
            # Resolve target entity
            if effect_selection_type_component is None:
                target_entity_ids = _handle_effect_selection_component_none(
                    query_target_entity_ids
                )

            elif effect_selection_type_component.value == EffectSelectionType.ALL:
                target_entity_ids = _handle_effect_selection_type_all(query_target_entity_ids)

            elif effect_selection_type_component.value == EffectSelectionType.RANDOM:
                target_entity_ids = _handle_effect_selection_type_random(query_target_entity_ids)

            elif effect_selection_type_component.value == EffectSelectionType.SPECIFIC:
                target_entity_ids = _handle_effect_selection_type_specific(
                    query_target_entity_ids, effect_entity_id, manager
                )
                if target_entity_ids is None:
                    # Early return, need to wait until target entities are selected
                    return

            # Tag target entities
            for target_entity_id in target_entity_ids:
                manager.add_component(target_entity_id, EffectTargetComponent())

        # Promote effect to targeted status
        manager.remove_component(effect_entity_id, EffectIsDispatchedComponent)
        manager.add_component(effect_entity_id, EffectIsTargetedComponent())
