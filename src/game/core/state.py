from enum import Enum


class GameState(Enum):
    BATTLE = "BATTLE"
    SHOP = "SHOP"
    EVENT = "EVENT"
    REWARD = "REWARD"
    MAP = "MAP"


class BattleState(Enum):
    DEFAULT = "DEFAULT"
    AWAIT_TARGET = "AWAIT_TARGET"
    NONE = "NONE"
