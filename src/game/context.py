from enum import Enum
from typing import List, Optional

from game.core.char import Character
from game.core.energy import Energy
from game.core.entity import Health
from game.core.monster import Monster
from game.lib.char import char_lib
from game.lib.monster import monster_lib


# TODO: define elsewhere
class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1
    NONE = 2


# TODO: parametrize
CHAR_NAME = "Silent"
MONSTER_NAME = "Dummy"

# Battle state
state = BattleState.NONE

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
active_card_idx: Optional[int] = None

# TODO: intialize elsewhere
# TODO: add modifiers
# Character
char = Character(name=CHAR_NAME, health=Health(char_lib[CHAR_NAME].base_health))

# Monsters. The monsters are implemented as a list of monsters
monsters = [Monster(name=MONSTER_NAME, health=Health(monster_lib[MONSTER_NAME].base_health))]
