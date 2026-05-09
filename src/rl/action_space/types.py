"""Head-type taxonomy and model-output → slai action conversion.

The HeadTypePrimary enum is kept identical to pre-migration so the model
heads' tensor layouts don't move. Routing policy (which head fires for a
given env state) lives in route.py; this file is just the conversion from
(head_type, primary_idx, selection_idx) to a slai action (or a buffering
marker for the two-step card-targeting wrapper).
"""

from enum import IntEnum

import slai

from src.rl.env_wrapper import _PendingCardPlay
from src.rl.env_wrapper import _ResolveCardPlay


class HeadTypePrimary(IntEnum):
    """
    Primary head types, one per FSM state that requires a decision.

    Decision primaries (binary choice + optional secondary selection):
        COMBAT_DEFAULT: [end_turn, play_card] → if play_card, run CARD_PLAY
        CARD_REWARD: [skip, select] → if select, run CARD_REWARD_SELECT
        REST_SITE: [rest, upgrade] → if upgrade, run CARD_UPGRADE

    Direct primaries (no primary choice, go straight to entity selection):
        COMBAT_CARD_DISCARD: pick card to discard
        COMBAT_MONSTER_SELECT: pick monster to target (driven by EnvWrapper
            re-presentation when a buffered card-play is awaiting a target)
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
# Primary head classification (list-based for fast int-indexed lookup)
# =========================================================================

NUM_PRIMARY_HEADS: int = len(HeadTypePrimary)

# IS_DECISION_PRIMARY[int(htp)] → True if this is a decision primary
IS_DECISION_PRIMARY: tuple[bool, ...] = tuple(
    htp
    in {
        HeadTypePrimary.COMBAT_DEFAULT,
        HeadTypePrimary.CARD_REWARD,
        HeadTypePrimary.REST_SITE,
    }
    for htp in HeadTypePrimary
)

# PRIMARY_NUM_CHOICES[int(htp)] → number of choices (0 for direct primaries)
PRIMARY_NUM_CHOICES: tuple[int, ...] = tuple(
    {
        HeadTypePrimary.COMBAT_DEFAULT: 2,
        HeadTypePrimary.CARD_REWARD: 2,
        HeadTypePrimary.REST_SITE: 2,
    }.get(htp, 0)
    for htp in HeadTypePrimary
)


# =========================================================================
# Model output → game Action conversion
# =========================================================================


def to_action(
    head_type_primary: HeadTypePrimary,
    primary_index: int,
    selection_index: int,
):
    """
    Convert model output to a slai action (or a buffering marker for
    two-step card-targeting).

    Returns either a `slai.Action.*` instance (passed straight to
    `EnvWrapper.step`) or a `_PendingCardPlay` / `_ResolveCardPlay` marker
    that the wrapper interprets.

    Args:
        head_type_primary: Which primary group this sample belongs to
        primary_index: Index chosen by the primary head (-1 for direct primaries)
        selection_index: Index chosen by the selection head (-1 if terminal)
    """
    match head_type_primary:
        # Decision primaries: primary_index 0 = terminal, 1 = select
        case HeadTypePrimary.COMBAT_DEFAULT:
            if primary_index == 0:
                return slai.Action.EndTurn()
            return _PendingCardPlay(idx_hand=selection_index)

        case HeadTypePrimary.CARD_REWARD:
            if primary_index == 0:
                return slai.Action.CardRewardSkip()
            return slai.Action.CardRewardSelect(idx_reward=selection_index)

        case HeadTypePrimary.REST_SITE:
            if primary_index == 0:
                return slai.Action.RestSiteRest()
            return slai.Action.RestSiteCardUpgrade(idx_deck=selection_index)

        # Direct primaries
        case HeadTypePrimary.COMBAT_CARD_DISCARD:
            # slai's CardDiscard takes a list of indices; the trainer picks
            # one card per model decision, so we wrap the single index.
            return slai.Action.CardDiscard(indices_hand=[selection_index])

        case HeadTypePrimary.COMBAT_MONSTER_SELECT:
            return _ResolveCardPlay(idx_monster=selection_index)

        case HeadTypePrimary.MAP_SELECT:
            return slai.Action.RoomSelect(idx_column=selection_index)

        case _:
            raise ValueError(f"Unknown head type primary: {head_type_primary}")
