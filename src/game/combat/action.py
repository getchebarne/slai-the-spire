from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    SELECT_ENTITY = "SELECT_ENTITY"
    END_TURN = "END_TURN"


@dataclass
class Action:
    type: ActionType
    target_id: Optional[int] = None
