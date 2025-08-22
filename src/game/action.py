from dataclasses import dataclass
from enum import Enum


class ActionType(Enum):
    # Card reward
    CARD_REWARD_SELECT = "CARD_REWARD_SELECT"
    CARD_REWARD_SKIP = "CARD_REWARD_SKIP"

    # Combat
    COMBAT_CARD_IN_HAND_SELECT = "COMBAT_CARD_IN_HAND_SELECT"
    COMBAT_MONSTER_SELECT = "COMBAT_MONSTER_SELECT"
    COMBAT_TURN_END = "COMBAT_TURN_END"

    # Map
    MAP_NODE_SELECT = "MAP_NODE_SELECT"

    # Rest site TODO: dig, lift, etc.
    REST_SITE_REST = "REST_SITE_REST"
    REST_SITE_UPGRADE = "REST_SITE_UPGRADE"


@dataclass
class Action:
    type: ActionType

    # Used to optionally determine the index of an action, e.g., the position of the card in the
    # hand to select, the monster to target, the map node to select, etc.
    index: int | None = None
