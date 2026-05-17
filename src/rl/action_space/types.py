"""Head-type taxonomy and model-output → slai action conversion.

Post-migration HeadTypePrimary slots (9 total). Order is stable —
existing checkpoints break on this version anyway because new heads have
new parameters. Don't reorder without bumping the model version.

Card-targeting is now inline: when COMBAT_DEFAULT picks "play_card" and
the chosen card requires a target, the same forward pass also picks the
monster. There is no separate COMBAT_MONSTER_SELECT primary.
"""

from enum import IntEnum

import slai
from slai import Action, ActionType


class HeadTypePrimary(IntEnum):
    """
    Primary head types, one per phase that requires a decision.

    Decision primaries (binary skip/take + optional secondary selection):
        COMBAT_DEFAULT   [end_turn,    play_card]    → if play_card: HeadCardPlay (+ inline target)
        CARD_REWARD      [skip,        select]       → if select:    HeadCardRewardSelect
        REST_SITE        [rest,        upgrade]      → if upgrade:   HeadCardUpgrade
        RELIC_REWARD     [skip,        select]       → if select:    HeadRelicSelect

    Direct primaries (no binary, head fires immediately):
        COMBAT_CARD_DISCARD     HeadCardMultiPick    (multi-pick over hand)
        COMBAT_AWAIT_RETAIN     HeadCardMultiPick    (multi-pick over hand)
        COMBAT_AWAIT_NIGHTMARE  HeadCardNightmare    (single hand pick)
        COMBAT_AWAIT_SETUP      HeadCardSetup        (single hand pick)
        MAP_SELECT              HeadMapSelect        (single column pick)
    """

    CARD_REWARD = 0
    COMBAT_CARD_DISCARD = 1
    COMBAT_DEFAULT = 2
    MAP_SELECT = 3
    REST_SITE = 4
    COMBAT_AWAIT_RETAIN = 5
    COMBAT_AWAIT_NIGHTMARE = 6
    COMBAT_AWAIT_SETUP = 7
    RELIC_REWARD = 8


# =========================================================================
# Primary head classification (list-based for fast int-indexed lookup)
# =========================================================================

NUM_PRIMARY_HEADS: int = len(HeadTypePrimary)

_DECISION_PRIMARIES = {
    HeadTypePrimary.COMBAT_DEFAULT,
    HeadTypePrimary.CARD_REWARD,
    HeadTypePrimary.REST_SITE,
    HeadTypePrimary.RELIC_REWARD,
}

# IS_DECISION_PRIMARY[int(htp)] → True if this is a decision primary
IS_DECISION_PRIMARY: tuple[bool, ...] = tuple(
    htp in _DECISION_PRIMARIES for htp in HeadTypePrimary
)

# PRIMARY_NUM_CHOICES[int(htp)] → number of choices (0 for direct primaries)
PRIMARY_NUM_CHOICES: tuple[int, ...] = tuple(
    2 if htp in _DECISION_PRIMARIES else 0 for htp in HeadTypePrimary
)


# =========================================================================
# Model output → slai action conversion
# =========================================================================


def to_action(
    head_type_primary: HeadTypePrimary,
    primary_index: int,
    selection_index: int,
    target_index: int = -1,
    retain_indices: list[int] | None = None,
):
    """
    Convert model output to a `slai.Action` instance with the appropriate
    `ActionType` discriminant and positional `indices`.

    Args:
        head_type_primary: Which primary group this sample belongs to.
        primary_index: Index from the binary decision head (-1 for direct primaries,
                       0 = terminal, 1 = select).
        selection_index: Index from the per-entity selection head (-1 if terminal).
        target_index: For COMBAT_DEFAULT play-card, the chosen monster idx (or -1
                      if the card needs no target).
        retain_indices: For COMBAT_AWAIT_RETAIN, the list of hand indices to retain
                        (already truncated to `num`).
    """
    match head_type_primary:
        case HeadTypePrimary.COMBAT_DEFAULT:
            if primary_index == 0:
                return Action(ActionType.EndTurn, [])
            indices = [selection_index]
            if target_index >= 0:
                indices.append(target_index)
            return Action(ActionType.CardPlay, indices)

        case HeadTypePrimary.CARD_REWARD:
            if primary_index == 0:
                return Action(ActionType.CardRewardSkip, [])
            return Action(ActionType.CardRewardSelect, [selection_index])

        case HeadTypePrimary.REST_SITE:
            if primary_index == 0:
                return Action(ActionType.RestSiteRest, [])
            return Action(ActionType.RestSiteCardUpgrade, [selection_index])

        case HeadTypePrimary.RELIC_REWARD:
            if primary_index == 0:
                return Action(ActionType.RelicRewardSkip, [])
            return Action(ActionType.RelicRewardSelect, [selection_index])

        case HeadTypePrimary.COMBAT_CARD_DISCARD:
            # CardDiscard requires exactly `num` indices (matching
            # CombatAwaitDiscard.num). Multi-pick output is in
            # `retain_indices` (shared multi-pick storage).
            assert retain_indices is not None, \
                "COMBAT_CARD_DISCARD requires retain_indices (shared multi-pick storage)"
            return Action(ActionType.CardDiscard, retain_indices)

        case HeadTypePrimary.COMBAT_AWAIT_NIGHTMARE:
            return Action(ActionType.CardNightmare, [selection_index])

        case HeadTypePrimary.COMBAT_AWAIT_SETUP:
            return Action(ActionType.CardSetup, [selection_index])

        case HeadTypePrimary.COMBAT_AWAIT_RETAIN:
            assert retain_indices is not None, \
                "COMBAT_AWAIT_RETAIN requires retain_indices"
            return Action(ActionType.CardRetain, retain_indices)

        case HeadTypePrimary.MAP_SELECT:
            return Action(ActionType.RoomSelect, [selection_index])

        case _:
            raise ValueError(f"Unknown head type primary: {head_type_primary}")
