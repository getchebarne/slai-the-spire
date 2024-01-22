from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from game.lib.char import char_lib
from game.lib.monster import monster_lib


@dataclass
class Energy:
    max: int
    current: int


class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1
    NONE = 2


@dataclass
class Entity:
    name: str
    max_health: int
    current_health: int
    block: int = 0


@dataclass
class Character(Entity):
    pass


@dataclass
class Monster(Entity):
    current_move_name: Optional[str] = None


CHAR_NAME = "Silent"
MONSTER_NAME = "Dummy"

# Cards
deck: List[str] = [
    "Strike",
    "Strike",
    "Strike",
    "Strike",
    "Strike",
    "Defend",
    "Defend",
    "Defend",
    "Defend",
    "Defend",
    "Neutralize",
]
hand: List[str] = []
draw_pile: List[str] = []
disc_pile: List[str] = []

# Energy
energy = Energy(max=3, current=3)

# Active card
active_card: Optional[str] = None

# TODO: intialize elsewhere
# TODO: add modifiers
# Character
char = Character(
    name=CHAR_NAME,
    max_health=char_lib[CHAR_NAME].base_health,
    current_health=char_lib[CHAR_NAME].base_health,
)
# Monsters
monsters = [
    Monster(
        name=MONSTER_NAME,
        max_health=monster_lib[MONSTER_NAME].base_health,
        current_health=monster_lib[MONSTER_NAME].base_health,
    )
]
# Battle state
state = BattleState.NONE
