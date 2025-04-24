from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    # Combat
    ENTITY_SELECT = "ENTITY_SELECT"
    TURN_END = "TURN_END"

    # Rest site TODO: dig, lift, etc.
    REST_SITE_REST = "REST_SITE_REST"
    REST_SITE_UPGRADE = "REST_SITE_UPGRADE"


@dataclass
class Action:
    type: ActionType
    target_id: int | None = None
