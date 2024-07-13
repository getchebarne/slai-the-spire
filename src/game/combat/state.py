from enum import Enum

from src.game.combat.entities import Entities


class State(Enum):
    DEFAULT = "DEFAULT"
    AWAIT_CARD_TARGET = "AWAIT_CARD_TARGET"
    AWAIT_EFFECT_TARGET = "AWAIT_EFFECT_TARGET"


def on_enter(state: State, entities: Entities) -> None:
    if state == State.DEFAULT:
        entities.card_active_id = None
        entities.card_target_id = None
        entities.effect_target_id = None
        entities.entity_selectable_ids = [
            card_in_hand_id
            for card_in_hand_id in entities.card_in_hand_ids
            if entities.get_entity(card_in_hand_id).cost
            <= entities.get_entity(entities.energy_id).current
        ]

    elif state == State.AWAIT_CARD_TARGET:
        entities.entity_selectable_ids = entities.monster_ids  # TODO: copy?
