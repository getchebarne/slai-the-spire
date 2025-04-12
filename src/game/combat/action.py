from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    SELECT_ENTITY = "SELECT_ENTITY"
    END_TURN = "END_TURN"


@dataclass
class Action:
    type: ActionType
    target_id: int | None = None
