from enum import Enum, IntEnum

from src.game.action import ActionType


class HeadType(Enum):
    """Secondary head types for entity selection."""

    CARD_PLAY = "CARD_PLAY"
    CARD_DISCARD = "CARD_DISCARD"
    CARD_REWARD_SELECT = "CARD_REWARD_SELECT"
    CARD_UPGRADE = "CARD_UPGRADE"
    MONSTER_SELECT = "MONSTER_SELECT"
    MAP_SELECT = "MAP_SELECT"


class ActionChoice(IntEnum):
    """
    Unambiguous action choices for the primary head.

    Unlike ActionType (where COMBAT_CARD_IN_HAND_SELECT is used for both
    play and discard), ActionChoice distinguishes all cases, enabling
    direct routing to secondary heads without FSM context.
    """

    # Terminal actions (no secondary head needed)
    COMBAT_TURN_END = 0
    CARD_REWARD_SKIP = 1
    REST_SITE_REST = 2

    # Actions requiring secondary heads
    CARD_PLAY = 3
    CARD_DISCARD = 4
    MONSTER_SELECT = 5
    CARD_REWARD_SELECT = 6
    CARD_UPGRADE = 7
    MAP_SELECT = 8


NUM_ACTION_CHOICES = len(ActionChoice)


# ActionChoice → HeadType (None for terminal actions)
CHOICE_TO_HEAD: dict[ActionChoice, HeadType | None] = {
    ActionChoice.COMBAT_TURN_END: None,
    ActionChoice.CARD_REWARD_SKIP: None,
    ActionChoice.REST_SITE_REST: None,
    ActionChoice.CARD_PLAY: HeadType.CARD_PLAY,
    ActionChoice.CARD_DISCARD: HeadType.CARD_DISCARD,
    ActionChoice.MONSTER_SELECT: HeadType.MONSTER_SELECT,
    ActionChoice.CARD_REWARD_SELECT: HeadType.CARD_REWARD_SELECT,
    ActionChoice.CARD_UPGRADE: HeadType.CARD_UPGRADE,
    ActionChoice.MAP_SELECT: HeadType.MAP_SELECT,
}


# ActionChoice → ActionType (for game interface)
CHOICE_TO_ACTION_TYPE: dict[ActionChoice, ActionType] = {
    ActionChoice.COMBAT_TURN_END: ActionType.COMBAT_TURN_END,
    ActionChoice.CARD_REWARD_SKIP: ActionType.CARD_REWARD_SKIP,
    ActionChoice.REST_SITE_REST: ActionType.REST_SITE_REST,
    ActionChoice.CARD_PLAY: ActionType.COMBAT_CARD_IN_HAND_SELECT,
    ActionChoice.CARD_DISCARD: ActionType.COMBAT_CARD_IN_HAND_SELECT,
    ActionChoice.MONSTER_SELECT: ActionType.COMBAT_MONSTER_SELECT,
    ActionChoice.CARD_REWARD_SELECT: ActionType.CARD_REWARD_SELECT,
    ActionChoice.CARD_UPGRADE: ActionType.REST_SITE_UPGRADE,
    ActionChoice.MAP_SELECT: ActionType.MAP_NODE_SELECT,
}
