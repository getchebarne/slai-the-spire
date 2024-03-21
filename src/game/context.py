from enum import Enum
from typing import Optional

import pandas as pd

from game.core.energy import Energy
from game.lib.char import char_lib
from game.lib.monster import monster_lib


# TODO: parametrize
CHAR_NAME = "Silent"
MONSTER_NAME = "Dummy"


# TODO: define elsewhere
class BattleState(Enum):
    DEFAULT = 0
    AWAIT_TARGET = 1
    NONE = 2


# TODO: define elsewhere
def monster_entity_ids() -> list[int]:
    return entities[~entities["entity_is_char"]].index.tolist()


# TODO: define elsewhere
def char_entity_id() -> int:
    id_ = entities[entities["entity_is_char"]].index
    if len(id_) != 1:
        raise ValueError("There should be exactly one character entity")

    return id_[0]


# Battle state
state = BattleState.NONE

# Cards
deck: list[str] = [
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
hand: list[str] = []
draw_pile: list[str] = []
disc_pile: list[str] = []

# Energy
energy = Energy(max=3, current=3)

# Active card
active_card_idx: Optional[int] = None

# TODO: intialize elsewhere
# TODO: add modifiers

# Entities
entities = pd.DataFrame(
    {
        "entity_id": [0, 1],
        "entity_name": [CHAR_NAME, MONSTER_NAME],
        "entity_current_health": [
            char_lib[CHAR_NAME].base_health,
            monster_lib[MONSTER_NAME].base_health,
        ],
        "entity_max_health": [
            char_lib[CHAR_NAME].base_health,
            monster_lib[MONSTER_NAME].base_health,
        ],
        "entity_current_block": [0, 0],
        "entity_is_char": [True, False],
    },
).set_index("entity_id")

# Entity modifiers
entities_modifiers = pd.DataFrame(
    {
        "entity_id": [],
        "modifier_name": [],
        "modifier_current_stacks": [],
    }
)
# Monster moves
monster_moves = pd.DataFrame(
    {
        "entity_id": [1],
        "current_move_name": None,
    }
).set_index("entity_id")
