import random
from enum import Enum
from typing import Optional

from src.game.combat.processors import get_effect_processors
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState
from src.game.combat.utils import add_effects_to_top


class EffectSelectionStatus(Enum):
    COMPLETE = "COMPLETE"
    PENDING_INPUT = "PENDING_INPUT"


def _resolve_effect_target_type(
    state: GameState, effect_target_type: EffectTargetType
) -> list[int]:
    if effect_target_type == EffectTargetType.CHARACTER:
        return [state.character_id]

    if effect_target_type == EffectTargetType.MONSTER:
        return state.monster_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [state.card_target_id]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return state.card_in_hand_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_ACTIVE:
        return [state.card_active_id]

    if effect_target_type == EffectTargetType.TURN:
        return [state.actor_turn_id]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    state: GameState, effect_selection_type: EffectSelectionType, entity_ids: list[int]
) -> tuple[EffectSelectionStatus, list[int]]:

    if effect_selection_type == EffectSelectionType.ALL:
        return EffectSelectionStatus.COMPLETE, entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        return EffectSelectionStatus.COMPLETE, [random.choice(entity_ids)]

    if effect_selection_type == EffectSelectionType.INPUT:
        if not state.selected_entity_ids:
            return EffectSelectionStatus.PENDING_INPUT, entity_ids

        return EffectSelectionStatus.COMPLETE, state.selected_entity_ids

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def get_effect_targets(
    state: GameState,
    effect_target_type: EffectTargetType,
    effect_selection_type: EffectSelectionType,
) -> tuple[Optional[EffectSelectionStatus], Optional[list[int]]]:
    if effect_target_type is None:
        return None, None

    query_entity_ids = _resolve_effect_target_type(state, effect_target_type)

    if effect_selection_type is None:
        return None, query_entity_ids

    return _resolve_effect_selection_type(state, effect_selection_type, query_entity_ids)


def _process_next_effect(state: GameState) -> Optional[EffectSelectionStatus]:
    # Get effect from queue
    effect, source_id = state.effect_queue.popleft()

    # Get effect's targets and processors
    effect_selection_status, target_ids = get_effect_targets(
        state, effect.target_type, effect.selection_type
    )
    if effect_selection_status == EffectSelectionStatus.PENDING_INPUT:
        # Tags
        state.effect_type = effect.type

        # Reque effect
        add_effects_to_top(state, (effect, source_id))

        return effect_selection_status

    processors = get_effect_processors(effect.type)
    state.effect_type = None
    state.selected_entity_ids = None

    # Execute
    if target_ids is None:
        target_ids = [None]

    state.effect_source_id = source_id
    for target_id in target_ids:
        # Set target and value
        state.effect_target_id = target_id
        state.effect_value = effect.value

        # Run effect's processors
        for processor in processors:
            processor(state)


def process_queue(state: GameState) -> None:
    ret = None
    while state.effect_queue and ret is None:
        ret = _process_next_effect(state)
