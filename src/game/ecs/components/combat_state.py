from dataclasses import dataclass
from enum import Enum

from src.game.ecs.components.base import BaseComponent


# TODO: revisit if turns needs to be separate from states
class CombatState(Enum):
    CHAR_TURN = "CHAR_TURN"
    AWAIT_CONFIRMATION = "AWAIT_CONFIRMATION"
    AWAIT_SELECT_TARGET = "SELECT_TARGET"
    PROCESSING_EFFECTS = "PROCESSING_EFFECTS"
    MONSTER_TURN = "MONSTER_TURN"


@dataclass
class CombatStateComponent(BaseComponent):
    value: CombatState
