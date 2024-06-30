from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    SELECT = "SELECT"
    CONFIRM = "CONFIRM"
    END_TURN = "END_TURN"


@dataclass
class Action:
    type: ActionType
    target_entity_id: Optional[int] = None
