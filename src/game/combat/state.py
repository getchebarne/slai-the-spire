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

    elif state == State.AWAIT_CARD_TARGET:
        entities.entitiy_selectable_ids = entities.monster_ids  # TODO: copy?
