from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    SELECT_CARD = "SELECT_CARD"
    SELECT_MONSTER = "SELECT_MONSTER"
    CONFIRM = "CONFIRM"
    END_TURN = "END_TURN"


@dataclass
class Action:
    type: ActionType
    target_entity_id: Optional[int] = None
