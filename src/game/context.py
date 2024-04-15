from enum import Enum
from dataclasses import dataclass
from typing import Optional, Generator

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


@dataclass
class EntityData:
    name: str
    current_health: int
    max_health: int
    current_block: int
    is_char: bool


# TODO: define elsewhere
def get_monsters() -> Generator[tuple[int, EntityData], None, None]:
    for entity_id, entity_data in entities.items():
        if not entity_data.is_char:
            yield entity_id, entity_data


# TODO: define elsewhere
def get_char() -> tuple[int, EntityData]:
    for entity_id, entity_data in entities.items():
        if entity_data.is_char:
            return entity_id, entity_data

    raise ValueError("No character found")


# TODO: intialize elsewhere
# TODO: add modifiers
entities = {
    0: EntityData(
        name=CHAR_NAME,
        current_health=char_lib[CHAR_NAME].base_health,
        max_health=char_lib[CHAR_NAME].base_health,
        current_block=0,
        is_char=True,
    ),
    1: EntityData(
        name=MONSTER_NAME,
        current_health=monster_lib[MONSTER_NAME].base_health,
        max_health=monster_lib[MONSTER_NAME].base_health,
        current_block=0,
        is_char=False,
    ),
}
entity_modifiers = {(0, "strength"): 1}
monster_moves = {1: None}
