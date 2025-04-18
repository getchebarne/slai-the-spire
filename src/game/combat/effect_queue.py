from __future__ import annotations

import random
from typing import TypeAlias

from src.game.combat.effect import EffectSelectionType
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import SourcedEffect
from src.game.combat.entities import EntityManager
from src.game.combat.processors import apply_effect


EffectQueue: TypeAlias = list[SourcedEffect]


def add_to_bot(effect_queue: EffectQueue, *sourced_effects: SourcedEffect) -> None:
    for sourced_effect in sourced_effects:
        effect_queue.append(sourced_effect)


def add_to_top(effect_queue: EffectQueue, *sourced_effects: SourcedEffect) -> None:
    for sourced_effect in reversed(sourced_effects):
        effect_queue.insert(0, sourced_effect)


class EffectNeedsInputTargets(Exception):
    pass


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType, entity_manager: EntityManager, id_source: int
) -> list[int]:
    if effect_target_type == EffectTargetType.SOURCE:
        return [id_source]

    if effect_target_type == EffectTargetType.CHARACTER:
        return [entity_manager.id_character]

    if effect_target_type == EffectTargetType.MONSTER:
        return entity_manager.id_monsters.copy()

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [entity_manager.id_card_target]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return entity_manager.id_cards_in_hand.copy()

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType, id_queries: list[int], id_effect_target: int | None
) -> list[int]:
    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(id_queries)]

    if effect_selection_type == EffectSelectionType.INPUT:
        # TODO: make more readable?
        if id_effect_target is None:
            # Verify if we need to prompt the player to select from query entities
            # or if no selection is needed
            # TODO: this can depend on the number of entities to select (e.g., "Prepared")
            num_target = 1
            if len(id_queries) > num_target:
                raise EffectNeedsInputTargets

            return id_queries

        return [id_effect_target]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def process_effect_queue(entity_manager: EntityManager, effect_queue: EffectQueue) -> None:
    while effect_queue:
        sourced_effect = effect_queue.pop(0)
        effect = sourced_effect.effect
        id_source = sourced_effect.id_source
        id_target = sourced_effect.id_target

        if id_target is None:
            if effect.target_type is None:
                # Assign a list with a single `None` target so the effect is applied once
                id_targets = [None]

            else:
                # Get effect's query entities
                id_queries = _resolve_effect_target_type(
                    effect.target_type, entity_manager, id_source
                )

                # Select from those entities
                if effect.selection_type is None:
                    id_targets = id_queries

                else:
                    try:
                        id_targets = _resolve_effect_selection_type(
                            effect.selection_type, id_queries, entity_manager.id_effect_target
                        )

                    except EffectNeedsInputTargets:
                        # Need to wait for player to select the effect's target. Put effect back
                        # into queue at position 0 and return id_queries to tag them as selectable
                        effect_queue.insert(0, sourced_effect)

                        return

        else:
            # TODO: could QueuedEffect have multiple target entities when created? think
            id_targets = [id_target]

        for id_target in id_targets:
            # Process the effect & get new effects to add to the queue
            sourced_effects_bot, sourced_effects_top = apply_effect(
                entity_manager, effect.type, effect.value, id_source, id_target
            )

            # Add new effects to the queue
            add_to_bot(effect_queue, *sourced_effects_bot)
            add_to_top(effect_queue, *sourced_effects_top)
