from enum import Enum


class HeadType(Enum):
    # Primary
    ACTION_TYPE = "ACTION_TYPE"

    # Card-related secondary
    CARD_PLAY = "CARD_PLAY"
    CARD_DISCARD = "CARD_DISCARD"
    CARD_REWARD_SELECT = "CARD_REWARD_SELECT"
    CARD_UPGRADE = "CARD_UPGRADE"

    # Other
    MONSTER_SELECT = "MONSTER_SELECT"
    MAP_SELECT = "MAP_SELECT"
