import random
from typing import Optional

from src.game.combat.processors import get_effect_processors
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType, state: GameState
) -> list[int]:
    if effect_target_type == EffectTargetType.CHARACTER:
        return [state.character_id]

    if effect_target_type == EffectTargetType.MONSTER:
        return state.monster_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [state.card_target_id]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return state.card_in_hand_ids.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.TURN:
        return [state.actor_turn_id]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType, entity_ids: list[int]
) -> list[int]:
    if effect_selection_type == effect_selection_type.SPECIFIC:
        # TODO: get action from agent
        pass

    if effect_selection_type == EffectSelectionType.ALL:
        return entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(entity_ids)]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def get_effect_targets(
    effect_target_type: EffectTargetType,
    effect_selection_type: EffectSelectionType,
    state: GameState,
) -> Optional[list[int]]:
    if effect_target_type is None:
        return None

    query_entity_ids = _resolve_effect_target_type(effect_target_type, state)

    if effect_selection_type is None:
        return query_entity_ids

    return _resolve_effect_selection_type(effect_selection_type, query_entity_ids)


def _process_next_effect(state: GameState) -> None:
    # Get effect from queue
    effect = state.effect_queue.popleft()

    # Get effect's targets and processors
    target_ids = get_effect_targets(effect.target_type, effect.selection_type, state)
    processors = get_effect_processors(effect.type)

    # Execute
    if target_ids is None:
        target_ids = [None]

    for target_id in target_ids:
        # Set target and value
        state.effect_target_id = target_id
        state.effect_value = effect.value

        # Run effect's processors
        for processor in processors:
            processor(state)


def process_queue(state: GameState) -> None:
    while state.effect_queue:
        _process_next_effect(state)
