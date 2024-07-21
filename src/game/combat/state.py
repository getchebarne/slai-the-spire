from enum import Enum
from typing import Optional

from src.game.combat.entities import Entities


class State(Enum):
    DEFAULT = "DEFAULT"
    AWAIT_CARD_TARGET = "AWAIT_CARD_TARGET"
    AWAIT_EFFECT_TARGET = "AWAIT_EFFECT_TARGET"


def on_enter(
    state: State, entities: Entities, entity_selectable_ids: Optional[list[int]] = None
) -> None:
    if state == State.DEFAULT:
        # Reset active card, card target, and effect target
        entities.card_active_id = None
        entities.card_target_id = None
        entities.effect_target_id = None

        # Set selectable entities to cards w/ cost lower or equal to the current energy
        entities.entity_selectable_ids = [
            card_in_hand_id
            for card_in_hand_id in entities.card_in_hand_ids
            if entities.get_entity(card_in_hand_id).cost
            <= entities.get_entity(entities.energy_id).current
        ]

        return

    if state == State.AWAIT_CARD_TARGET:
        entities.entity_selectable_ids = entities.monster_ids  # TODO: copy?

        return

    if state == State.AWAIT_EFFECT_TARGET:
        if entity_selectable_ids is None:
            raise ValueError(f"Must specify `entity_selectable_ids` for state {state}")

        entities.entity_selectable_ids = entity_selectable_ids
