from __future__ import annotations

import random
from dataclasses import replace

from src.game.combat.entities import EffectSelectionType
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import Entities
from src.game.combat.processors import process_effect
from src.game.combat.state import QueuedEffect
from src.game.combat.state import add_to_bot
from src.game.combat.state import add_to_top


class EffectNeedsInputTargets(Exception):
    pass


def _resolve_effect_target_type(
    entities: Entities, source_id: int, effect_target_type: EffectTargetType | None
) -> list[int | None]:
    if effect_target_type is None:
        return [None]

    if effect_target_type == EffectTargetType.SOURCE:
        return [source_id]

    if effect_target_type == EffectTargetType.CHARACTER:
        return [entities.character_id]

    if effect_target_type == EffectTargetType.MONSTER:
        return entities.monster_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [entities.card_target_id]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return entities.card_in_hand_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_ACTIVE:
        return [entities.card_active_id]

    if effect_target_type == EffectTargetType.SOURCE:  # TODO: fix
        return [entities.actor_turn_id]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    entities: Entities, entity_ids: list[int] | None, effect_selection_type: EffectSelectionType
) -> list[int]:
    # TODO: check this case outsde this function
    if effect_selection_type is None:
        return entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        # TODO: add support for multiple random targets
        return [random.choice(entity_ids)]

    if effect_selection_type == EffectSelectionType.INPUT:
        # TODO: make more readable?
        if entities.effect_target_id is None:
            # TODO: this can depend on the number of entities to select (e.g., "Prepared")
            num_target = 1
            if len(entity_ids) > num_target:
                raise EffectNeedsInputTargets

            return entity_ids

        return [entities.effect_target_id]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


# TODO: improve
def process_queue(
    entities: Entities, effect_queue: list[QueuedEffect]
) -> tuple[Entities, list[QueuedEffect]]:
    effect_queue_new = effect_queue.copy()

    while effect_queue_new:
        queued_effect = effect_queue_new.pop(0)
        source_id = queued_effect.source_id
        effect = queued_effect.effect

        # Get effect's query entities
        query_ids = _resolve_effect_target_type(entities, source_id, effect.target_type)

        # Select from those entities
        try:
            target_ids = _resolve_effect_selection_type(entities, query_ids, effect.selection_type)

        except EffectNeedsInputTargets:
            # Put effect back into queue at position 0
            effect_queue_new.insert(0, queued_effect)
            return replace(entities, entity_selectable_ids=query_ids), effect_queue_new

        for target_id in target_ids:
            # Process the effect, get new entities & new effects to add to the queue
            entities, effects_bot, effects_top = process_effect(
                entities, source_id, target_id, effect
            )

            for queued_effect in effects_bot:
                effect_queue_new = add_to_bot(
                    effect_queue_new, queued_effect.source_id, queued_effect.effect
                )

            for queued_effect in effects_top:
                effect_queue_new = add_to_top(
                    effect_queue_new, queued_effect.source_id, queued_effect.effect
                )

    return entities, effect_queue_new
