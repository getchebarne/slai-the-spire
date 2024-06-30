from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    SELECT_CARD = "SELECT_CARD"
    SELECT_MONSTER = "SELECT_MONSTER"
    END_TURN = "END_TURN"


@dataclass
class Action:
    type: ActionType
    index: Optional[int] = None
