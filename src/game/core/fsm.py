from enum import Enum


class FSM(Enum):
    # Card reward
    CARD_REWARD = "CARD_REWARD"

    # Combat
    COMBAT_DEFAULT = "COMBAT_DEFAULT"
    COMBAT_AWAIT_TARGET_CARD = "COMBAT_AWAIT_TARGET_CARD"
    COMBAT_AWAIT_TARGET_DISCARD = "COMBAT_AWAIT_TARGET_DISCARD"  # TODO: handle multiple targets

    # Map
    MAP = "MAP"

    # Rest site
    REST_SITE = "REST_SITE"
