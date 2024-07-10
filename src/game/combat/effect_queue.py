from __future__ import annotations

import random
from typing import Optional

from src.game.combat.entities import Effect
from src.game.combat.entities import EffectSelectionType
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import Entities
from src.game.combat.processors import get_effect_processors


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
    entities: Entities, effect_target_type: Optional[EffectTargetType]
) -> Optional[list[int]]:
    if effect_target_type is None:
        return None

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

    if effect_target_type == EffectTargetType.TURN:
        return [entities.actor_turn_id]

    raise ValueError(f"Unsupported effect target type: {effect_target_type}")


def _resolve_effect_selection_type(
    entities: Entities, entity_ids: Optional[list[int]], effect_selection_type: EffectSelectionType
) -> list[int]:
    if effect_selection_type is None:
        return entity_ids

    if effect_selection_type == EffectSelectionType.RANDOM:
        return [random.choice(entity_ids)]

    if effect_selection_type == EffectSelectionType.INPUT:
        if not entities.effect_target_id:
            raise EffectNeedsInputTargetsException

        return [entities.effect_target_id]

    raise ValueError(f"Unsupported effect selection type: {effect_selection_type}")


# TODO: improve
def process_queue(entities: Entities, effect_queue: EffectQueue) -> None:
    for source_id, effect in effect_queue:
        # Get effect's query entities
        query_ids = _resolve_effect_target_type(entities, effect.target_type)

        # Select from those entities
        try:
            target_ids = _resolve_effect_selection_type(entities, query_ids, effect.selection_type)

        except EffectNeedsInputTargetsException:
            # Set selectable entities & stop processing
            entities.entity_selectable_ids = query_ids

            return

        effect_queue.clear_pending()

        # Clear selectable entities TODO: move
        entities.entity_selectable_ids = None
        entities.effect_target_id = None  # TODO: revisit nullable

        # TODO: can this be a bit nicer?
        if target_ids is None:
            target_ids = [None]

        for target_id in target_ids:
            for processor in get_effect_processors(effect.type):
                effects_bot, effects_top = processor(
                    entities, source_id=source_id, target_id=target_id, value=effect.value
                )

                for source_id, effect_bot in effects_bot:
                    effect_queue.add_to_bot(source_id, effect_bot)

                for source_id, effect_top in effects_top:
                    effect_queue.add_to_top(source_id, effect_top)
