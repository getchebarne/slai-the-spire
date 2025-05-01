from enum import Enum


class FSM(Enum):
    # Combat
    COMBAT_DEFAULT = "COMBAT_DEFAULT"
    COMBAT_AWAIT_TARGET_CARD = "COMBAT_AWAIT_TARGET_CARD"
    COMBAT_AWAIT_TARGET_DISCARD = "COMBAT_AWAIT_TARGET_DISCARD"  # TODO: handle multiple targets

    # Rest site
    REST_SITE = "REST_SITE"

    # Map
    MAP = "MAP"
