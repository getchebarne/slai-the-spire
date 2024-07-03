import random
from typing import Optional

from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import Entity
from src.game.combat.state import GameState
from src.game.combat.processors import get_effect_processors


def _resolve_effect_target_type(
    effect_target_type: EffectTargetType, context: GameState
) -> list[Entity]:
    if effect_target_type == EffectTargetType.CHARACTER:
        return [context.character]

    if effect_target_type == EffectTargetType.MONSTER:
        return context.monsters.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.CARD_TARGET:
        return [context.card_target]

    if effect_target_type == EffectTargetType.CARD_IN_HAND:
        return context.hand.copy()  # TODO: revisit copy call

    if effect_target_type == EffectTargetType.TURN:
        return [context.turn]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    effect_selection_type: EffectSelectionType, entities: list[Entity]
) -> list[Entity]:
    if effect_selection_type == effect_selection_type.SPECIFIC:
        # TODO: get action from agent
        pass

    if effect_selection_type == EffectSelectionType.ALL:
        return entities

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(entities)]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


def get_effect_targets(
    effect_target_type: EffectTargetType,
    effect_selection_type: EffectSelectionType,
    context: GameState,
) -> Optional[list[Entity]]:
    if effect_target_type is None:
        return None

    query_entity_ids = _resolve_effect_target_type(effect_target_type, context)

    if effect_selection_type is None:
        return query_entity_ids

    return _resolve_effect_selection_type(effect_selection_type, query_entity_ids)


def _process_next_effect(context: GameState) -> None:
    # Get effect from queue
    effect = context.effect_queue.popleft()

    # Get effect's targets and processors
    targets = get_effect_targets(effect.target_type, effect.selection_type, context)
    processors = get_effect_processors(effect.type)

    # Execute
    if targets is None:
        targets = [None]

    for target in targets:
        # Set target and value
        context.effect_target = target
        context.effect_value = effect.value

        # Run effect's processors
        for processor in processors:
            processor(context)


def process_queue(context: GameState) -> None:
    while context.effect_queue:
        _process_next_effect(context)
