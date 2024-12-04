from __future__ import annotations

import random

from src.game.combat.entities import Effect
from src.game.combat.entities import EffectSelectionType
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Entities
from src.game.combat.processors import EffectPayload
from src.game.combat.processors import get_effect_processors


class EffectNeedsInputTargets(Exception):
    pass


class EffectQueue:
    def __init__(self):
        self._source_id_effects: list[tuple[int, Effect]] = []

    def __len__(self) -> int:
        return len(self._source_id_effects)

    def __iter__(self) -> EffectQueue:
        return self

    def __next__(self) -> tuple[int, Effect]:
        if not self._source_id_effects:
            # No more effects to process
            raise StopIteration

        return self._source_id_effects.pop(0)

    def add_to_bot(self, source_id: int, *effects: Effect) -> None:
        self._source_id_effects += [(source_id, effect) for effect in effects]

    def add_to_top(self, source_id: int, *effects: Effect) -> None:
        self._source_id_effects = [
            (source_id, effect) for effect in effects
        ] + self._source_id_effects

    # TODO: revisit
    def next_effect_type(self) -> EffectType:
        return self._source_id_effects[0][1].type

    def __str__(self) -> str:
        if not self._source_id_effects:
            return "EffectQueue is empty."

        return "EffectQueue:\n" + "\n".join(
            f"  Source ID {source_id}: {effect}" for source_id, effect in self._source_id_effects
        )


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
def process_queue(entities: Entities, effect_queue: EffectQueue) -> list[int] | None:
    for source_id, effect in effect_queue:
        # Get effect's query entities
        query_ids = _resolve_effect_target_type(entities, source_id, effect.target_type)

        # Select from those entities
        try:
            target_ids = _resolve_effect_selection_type(entities, query_ids, effect.selection_type)

        except EffectNeedsInputTargets:
            # Add effect back to the top of the queue
            effect_queue.add_to_top(source_id, effect)

            # Return selectable entities (which are the effect's query entities)
            return query_ids

        for target_id in target_ids:
            # TODO: improve payload handling and definiton
            payload = EffectPayload(source_id, target_id, effect.value)
            for processor in get_effect_processors(effect.type):
                effects_bot, effects_top = processor(entities, payload)

                for source_id, effect_bot in effects_bot:
                    effect_queue.add_to_bot(source_id, effect_bot)

                for source_id, effect_top in effects_top:
                    effect_queue.add_to_top(source_id, effect_top)
