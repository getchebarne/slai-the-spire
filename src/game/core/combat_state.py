from enum import Enum


class CombatState(Enum):
    DEFAULT = "DEFAULT"
    AWAIT_TARGET_CARD = "AWAIT_TARGET_CARD"
    AWAIT_TARGET_EFFECT = "AWAIT_TARGET_EFFECT"
