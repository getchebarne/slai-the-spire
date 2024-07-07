from __future__ import annotations

import random
from typing import Optional

from src.game.combat.processors import get_effect_processors
from src.game.combat.state import Effect
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import GameState


class EffectNeedsInputTargetsException(Exception):
    pass


class EffectQueue:
    def __init__(self):
        self._source_id_effects: list[tuple[int, Effect]] = []
        self._source_id_effect_pending: Optional[tuple[int, Effect]] = None

    def __iter__(self) -> EffectQueue:
        # Return the iterator object (in this case, the instance itself)
        return self

    def __next__(self) -> tuple[int, Effect]:
        if self.is_empty():
            raise StopIteration  # No more effects to process

        if self._source_id_effect_pending is not None:
            return self._source_id_effect_pending

        if self._source_id_effects:
            self._source_id_effect_pending = self._source_id_effects.pop(0)
            return self._source_id_effect_pending

        raise StopIteration

    def add_to_bot(self, source_id: int, *effects: Effect) -> None:
        self._source_id_effects += [(source_id, effect) for effect in effects]

    def add_to_top(self, source_id: int, *effects: Effect) -> None:
        self._source_id_effects = [
            (source_id, effect) for effect in effects
        ] + self._source_id_effects

    def is_empty(self) -> bool:
        return self._source_id_effect_pending is None and not self._source_id_effects

    def get_pending(self) -> Optional[tuple[int, Effect]]:
        return self._source_id_effect_pending

    def clear_pending(self) -> None:
        self._source_id_effect_pending = None


def _resolve_effect_target_type(
    state: GameState, effect_target_type: Optional[EffectTargetType]
) -> Optional[list[int]]:
    if effect_target_type is None:
        return None

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
    state: GameState, entity_ids: Optional[list[int]], effect_selection_type: EffectSelectionType
) -> list[int]:
    if effect_selection_type is None:
        return entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(entity_ids)]

    if effect_selection_type == EffectSelectionType.INPUT:
        if not state.entity_selected_ids:
            raise EffectNeedsInputTargetsException

        return state.entity_selected_ids

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


# TODO: improve
def process_queue(state: GameState, effect_queue: EffectQueue) -> None:
    for source_id, effect in effect_queue:
        # Get effect's query entities
        query_ids = _resolve_effect_target_type(state, effect.target_type)

        # Select from those entities
        try:
            target_ids = _resolve_effect_selection_type(state, query_ids, effect.selection_type)

        except EffectNeedsInputTargetsException:
            # Set selectable entities & stop processing
            state.entity_selectable_ids = query_ids

            return

        effect_queue.clear_pending()

        # Clear selectable entities
        state.entity_selectable_ids = None

        # TODO: can this be a bit nicer?
        if target_ids is None:
            target_ids = [None]

        for target_id in target_ids:
            for processor in get_effect_processors(effect.type):
                effects_bot, effects_top = processor(
                    state, source_id=source_id, target_id=target_id, value=effect.value
                )

                for source_id, effect_bot in effects_bot:
                    effect_queue.add_to_bot(source_id, effect_bot)

                for source_id, effect_top in effects_top:
                    effect_queue.add_to_top(source_id, effect_top)
