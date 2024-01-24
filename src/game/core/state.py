from enum import Enum


class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1
    NONE = 2
