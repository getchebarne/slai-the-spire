from enum import IntEnum

from src.game.action import Action
from src.game.action import ActionType


class HeadTypePrimary(IntEnum):
    """
    Primary head types, one per FSM state that requires a decision.

    Decision primaries (binary choice + optional secondary selection):
        COMBAT_DEFAULT: [end_turn, play_card] → if play_card, run CARD_PLAY
        CARD_REWARD: [skip, select] → if select, run CARD_REWARD_SELECT
        REST_SITE: [rest, upgrade] → if upgrade, run CARD_UPGRADE

    Direct primaries (no primary choice, go straight to entity selection):
        COMBAT_CARD_DISCARD: pick card to discard
        COMBAT_MONSTER_SELECT: pick monster to target
        MAP_SELECT: pick map node
    """

    CARD_REWARD = 0

    COMBAT_CARD_DISCARD = 1
    COMBAT_DEFAULT = 2
    COMBAT_MONSTER_SELECT = 3

    MAP_SELECT = 4

    REST_SITE = 5


class HeadTypeSecondary(IntEnum):
    """Secondary heads triggered by decision primaries."""

    CARD_PLAY = 0
    CARD_REWARD_SELECT = 1
    CARD_UPGRADE = 2


# =========================================================================
# Primary head classification
# =========================================================================

# Decision primaries: binary choice, may trigger a secondary head
DECISION_PRIMARIES: frozenset[HeadTypePrimary] = frozenset(
    {
        HeadTypePrimary.COMBAT_DEFAULT,
        HeadTypePrimary.CARD_REWARD,
        HeadTypePrimary.REST_SITE,
    }
)

# Direct primaries: go straight to entity selection (no binary choice)
DIRECT_PRIMARIES: frozenset[HeadTypePrimary] = frozenset(
    {
        HeadTypePrimary.COMBAT_CARD_DISCARD,
        HeadTypePrimary.COMBAT_MONSTER_SELECT,
        HeadTypePrimary.MAP_SELECT,
    }
)


# =========================================================================
# Decision primary → secondary head mapping
# =========================================================================

# For decision primaries: which secondary head to run when primary_index == 1
PRIMARY_TO_SECONDARY: dict[HeadTypePrimary, HeadTypeSecondary] = {
    HeadTypePrimary.COMBAT_DEFAULT: HeadTypeSecondary.CARD_PLAY,
    HeadTypePrimary.CARD_REWARD: HeadTypeSecondary.CARD_REWARD_SELECT,
    HeadTypePrimary.REST_SITE: HeadTypeSecondary.CARD_UPGRADE,
}

# Number of choices for each decision primary (all binary for now)
PRIMARY_NUM_CHOICES: dict[HeadTypePrimary, int] = {
    HeadTypePrimary.COMBAT_DEFAULT: 2,  # [end_turn, play_card]
    HeadTypePrimary.CARD_REWARD: 2,  # [skip, select]
    HeadTypePrimary.REST_SITE: 2,  # [rest, upgrade]
}


# =========================================================================
# Model output → game Action conversion
# =========================================================================


def to_action(
    head_type_primary: HeadTypePrimary,
    primary_index: int,
    selection_index: int,
) -> Action:
    """
    Convert model output to game Action.

    Args:
        head_type_primary: Which primary group this sample belongs to
        primary_index: Index chosen by the primary head (-1 for direct primaries)
        selection_index: Index chosen by the selection head (-1 if terminal)
    """
    match head_type_primary:
        # Decision primaries: primary_index 0 = terminal, 1 = select
        case HeadTypePrimary.COMBAT_DEFAULT:
            if primary_index == 0:
                return Action(type=ActionType.COMBAT_TURN_END)
            return Action(type=ActionType.COMBAT_CARD_IN_HAND_SELECT, index=selection_index)

        case HeadTypePrimary.CARD_REWARD:
            if primary_index == 0:
                return Action(type=ActionType.CARD_REWARD_SKIP)
            return Action(type=ActionType.CARD_REWARD_SELECT, index=selection_index)

        case HeadTypePrimary.REST_SITE:
            if primary_index == 0:
                return Action(type=ActionType.REST_SITE_REST)
            return Action(type=ActionType.REST_SITE_UPGRADE, index=selection_index)

        # Direct primaries: no primary decision, selection_index is the action
        case HeadTypePrimary.COMBAT_CARD_DISCARD:
            return Action(type=ActionType.COMBAT_CARD_IN_HAND_SELECT, index=selection_index)

        case HeadTypePrimary.COMBAT_MONSTER_SELECT:
            return Action(type=ActionType.COMBAT_MONSTER_SELECT, index=selection_index)

        case HeadTypePrimary.MAP_SELECT:
            return Action(type=ActionType.MAP_NODE_SELECT, index=selection_index)

        case _:
            raise ValueError(f"Unknown head type primary: {head_type_primary}")
