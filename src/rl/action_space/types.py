"""Head-type taxonomy + model-output → slai action conversion.

The engine's `recompute_legal_actions` is the source of truth this mirrors:

  - `game_over`                      → no actions
  - `pending is not None`            → ONLY the pending card-pick (no potions)
  - else dispatch on `screen`        → screen-native actions + potion use/discard

So routing is by `(pending-effect-class | screen)`, and every NON-pending screen
head is a categorical over that screen's legal *action-kinds* (OptKind), including
USE_POTION / DISCARD_POTION (the engine appends `push_potion_actions` to every
screen). An option-kind may carry a secondary entity selection and, for
CARD_PLAY / USE_POTION, a tertiary inline monster target.

Pending heads are direct: a multi-pick (discard/retain) or a single select
(setup/nightmare/discover/deck-pick).
"""

from enum import IntEnum

from slai import Action
from slai import ActionType


# =============================================================================
# Primary head taxonomy (screen heads + pending-pick heads)
# =============================================================================


class HeadTypePrimary(IntEnum):
    # Screen heads (option-kind categorical + nested selection)
    COMBAT = 0
    MAP = 1
    REST = 2
    REWARD = 3
    SHOP = 4
    EVENT = 5
    CHEST = 6
    # Pending multi-pick heads (count from pending Input)
    PEND_DISCARD = 7
    PEND_RETAIN = 8
    # Pending single-select heads
    PEND_SETUP = 9
    PEND_NIGHTMARE = 10
    PEND_DISCOVER = 11
    PEND_DECK_PURGE = 12
    PEND_DECK_UPGRADE = 13
    PEND_DECK_DUPLICATE = 14
    PEND_DECK_TRANSFORM = 15


NUM_PRIMARY_HEADS: int = len(HeadTypePrimary)


# =============================================================================
# Option kinds (the per-screen categorical choices) and selection contexts
# =============================================================================


class OptKind(IntEnum):
    TURN_END = 0
    CARD_PLAY = 1
    USE_POTION = 2
    DISCARD_POTION = 3
    ROOM_SELECT = 4
    REST = 5
    REST_UPGRADE = 6
    ROOM_EXIT = 7
    CHEST_OPEN = 8
    REWARD_TAKE_CARD = 9
    REWARD_TAKE_RELIC = 10
    REWARD_TAKE_POTION = 11
    REWARD_TAKE_GOLD = 12
    SHOP_BUY_CARD = 13
    SHOP_BUY_RELIC = 14
    SHOP_BUY_POTION = 15
    SHOP_PURGE = 16
    EVENT_OPTION = 17


class SelKey(IntEnum):
    """A distinct entity-selection head. Each maps to a pool source tensor +
    a per-sample selection mask (see masks.py SEL_SPECS)."""

    CARD_PLAY = 0
    POTION_USE = 1
    POTION_DISCARD = 2
    ROOM_SELECT = 3
    REST_UPGRADE = 4
    REWARD_CARD = 5
    SHOP_CARD = 6
    SHOP_RELIC = 7
    SHOP_POTION = 8
    SHOP_PURGE = 9
    EVENT_OPTION = 10
    PEND_SETUP = 11
    PEND_NIGHTMARE = 12
    PEND_DISCOVER = 13
    PEND_PURGE = 14
    PEND_UPGRADE = 15
    PEND_DUPLICATE = 16
    PEND_TRANSFORM = 17


NUM_SEL_KEYS: int = len(SelKey)


# Per-option-kind: (selection key or None, needs monster target). USE_POTION /
# CARD_PLAY targeting is conditional (only when the chosen entity requires_target);
# the flag here marks that the option *may* target.
_OPT_META: dict[OptKind, tuple] = {
    OptKind.TURN_END: (None, False),
    OptKind.CARD_PLAY: (SelKey.CARD_PLAY, True),
    OptKind.USE_POTION: (SelKey.POTION_USE, True),
    OptKind.DISCARD_POTION: (SelKey.POTION_DISCARD, False),
    OptKind.ROOM_SELECT: (SelKey.ROOM_SELECT, False),
    OptKind.REST: (None, False),
    OptKind.REST_UPGRADE: (SelKey.REST_UPGRADE, False),
    OptKind.ROOM_EXIT: (None, False),
    OptKind.CHEST_OPEN: (None, False),
    OptKind.REWARD_TAKE_CARD: (SelKey.REWARD_CARD, False),
    OptKind.REWARD_TAKE_RELIC: (None, False),
    OptKind.REWARD_TAKE_POTION: (None, False),
    OptKind.REWARD_TAKE_GOLD: (None, False),
    OptKind.SHOP_BUY_CARD: (SelKey.SHOP_CARD, False),
    OptKind.SHOP_BUY_RELIC: (SelKey.SHOP_RELIC, False),
    OptKind.SHOP_BUY_POTION: (SelKey.SHOP_POTION, False),
    OptKind.SHOP_PURGE: (SelKey.SHOP_PURGE, False),
    OptKind.EVENT_OPTION: (SelKey.EVENT_OPTION, False),
}


def opt_sel_key(opt: OptKind) -> "SelKey | None":
    return _OPT_META[opt][0]


def opt_may_target(opt: OptKind) -> bool:
    return _OPT_META[opt][1]


# Per-screen ordered option-kind lists (the categorical). Order is STABLE —
# reordering invalidates trained checkpoints. Potion kinds are appended to every
# screen (the engine's push_potion_actions is unconditional).
SCREEN_OPTION_KINDS: dict[HeadTypePrimary, tuple] = {
    HeadTypePrimary.COMBAT: (
        OptKind.TURN_END,
        OptKind.CARD_PLAY,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.MAP: (
        OptKind.ROOM_SELECT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.REST: (
        OptKind.REST,
        OptKind.REST_UPGRADE,
        OptKind.ROOM_EXIT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.REWARD: (
        OptKind.REWARD_TAKE_CARD,
        OptKind.REWARD_TAKE_RELIC,
        OptKind.REWARD_TAKE_POTION,
        OptKind.REWARD_TAKE_GOLD,
        OptKind.ROOM_EXIT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.SHOP: (
        OptKind.SHOP_BUY_CARD,
        OptKind.SHOP_BUY_RELIC,
        OptKind.SHOP_BUY_POTION,
        OptKind.SHOP_PURGE,
        OptKind.ROOM_EXIT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.EVENT: (
        OptKind.EVENT_OPTION,
        OptKind.ROOM_EXIT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
    HeadTypePrimary.CHEST: (
        OptKind.CHEST_OPEN,
        OptKind.ROOM_EXIT,
        OptKind.USE_POTION,
        OptKind.DISCARD_POTION,
    ),
}

# Screen heads are those with an option categorical.
SCREEN_HEADS: tuple = tuple(SCREEN_OPTION_KINDS.keys())
IS_SCREEN_HEAD: tuple = tuple(htp in SCREEN_OPTION_KINDS for htp in HeadTypePrimary)
PRIMARY_NUM_CHOICES: tuple = tuple(
    len(SCREEN_OPTION_KINDS[htp]) if htp in SCREEN_OPTION_KINDS else 0 for htp in HeadTypePrimary
)

# Pending multi-pick heads (hand k-subset; k from pending Input.count).
MULTIPICK_HEADS: tuple = (HeadTypePrimary.PEND_DISCARD, HeadTypePrimary.PEND_RETAIN)
IS_MULTIPICK: tuple = tuple(htp in MULTIPICK_HEADS for htp in HeadTypePrimary)

# Pending single-select heads → their selection key.
PEND_SINGLE_SELKEY: dict[HeadTypePrimary, SelKey] = {
    HeadTypePrimary.PEND_SETUP: SelKey.PEND_SETUP,
    HeadTypePrimary.PEND_NIGHTMARE: SelKey.PEND_NIGHTMARE,
    HeadTypePrimary.PEND_DISCOVER: SelKey.PEND_DISCOVER,
    HeadTypePrimary.PEND_DECK_PURGE: SelKey.PEND_PURGE,
    HeadTypePrimary.PEND_DECK_UPGRADE: SelKey.PEND_UPGRADE,
    HeadTypePrimary.PEND_DECK_DUPLICATE: SelKey.PEND_DUPLICATE,
    HeadTypePrimary.PEND_DECK_TRANSFORM: SelKey.PEND_TRANSFORM,
}


# =============================================================================
# Model output → slai action conversion
# =============================================================================


def to_action(
    htp: HeadTypePrimary,
    option_index: int,
    selection_index: int,
    target_index: int = -1,
    retain_indices: "list[int] | None" = None,
) -> Action:
    """Convert a model decision into a `slai.Action`.

    Args:
        htp: routed primary head.
        option_index: index into SCREEN_OPTION_KINDS[htp] (screen heads); -1 for
            pending heads.
        selection_index: chosen entity index (-1 if the option/head takes none).
        target_index: chosen alive-monster index for CARD_PLAY / USE_POTION
            targeting (-1 if untargeted).
        retain_indices: hand indices for PEND_DISCARD / PEND_RETAIN multi-pick.
    """
    # Pending multi-pick
    if htp == HeadTypePrimary.PEND_DISCARD:
        assert retain_indices is not None
        return Action(ActionType.CardDiscard, retain_indices)
    if htp == HeadTypePrimary.PEND_RETAIN:
        assert retain_indices is not None
        return Action(ActionType.CardRetain, retain_indices)
    # Pending single-select
    if htp == HeadTypePrimary.PEND_SETUP:
        return Action(ActionType.CardSetup, [selection_index])
    if htp == HeadTypePrimary.PEND_NIGHTMARE:
        return Action(ActionType.CardNightmare, [selection_index])
    if htp == HeadTypePrimary.PEND_DISCOVER:
        return Action(ActionType.CardDiscover, [selection_index])
    if htp == HeadTypePrimary.PEND_DECK_PURGE:
        return Action(ActionType.CardPurge, [selection_index])
    if htp == HeadTypePrimary.PEND_DECK_UPGRADE:
        return Action(ActionType.CardUpgrade, [selection_index])
    if htp == HeadTypePrimary.PEND_DECK_DUPLICATE:
        return Action(ActionType.CardDuplicate, [selection_index])
    if htp == HeadTypePrimary.PEND_DECK_TRANSFORM:
        return Action(ActionType.CardTransform, [selection_index])

    # Screen heads: resolve the chosen option-kind
    opt = SCREEN_OPTION_KINDS[htp][option_index]

    if opt == OptKind.TURN_END:
        return Action(ActionType.TurnEnd, [])
    if opt == OptKind.ROOM_EXIT:
        return Action(ActionType.RoomExit, [])
    if opt == OptKind.REST:
        return Action(ActionType.Rest, [])
    if opt == OptKind.CHEST_OPEN:
        return Action(ActionType.ChestOpen, [])
    if opt == OptKind.REWARD_TAKE_RELIC:
        return Action(ActionType.RewardTakeRelic, [])
    if opt == OptKind.REWARD_TAKE_POTION:
        return Action(ActionType.RewardTakePotion, [])
    if opt == OptKind.REWARD_TAKE_GOLD:
        return Action(ActionType.RewardTakeGold, [])

    if opt == OptKind.CARD_PLAY:
        idxs = [selection_index]
        if target_index >= 0:
            idxs.append(target_index)
        return Action(ActionType.CardPlay, idxs)
    if opt == OptKind.USE_POTION:
        idxs = [selection_index]
        if target_index >= 0:
            idxs.append(target_index)
        return Action(ActionType.PotionUse, idxs)
    if opt == OptKind.DISCARD_POTION:
        return Action(ActionType.PotionDiscard, [selection_index])
    if opt == OptKind.ROOM_SELECT:
        return Action(ActionType.RoomSelect, [selection_index])
    if opt == OptKind.REST_UPGRADE:
        return Action(ActionType.CardUpgrade, [selection_index])
    if opt == OptKind.REWARD_TAKE_CARD:
        return Action(ActionType.RewardTakeCard, [selection_index])
    if opt == OptKind.SHOP_BUY_CARD:
        return Action(ActionType.ShopBuyCard, [selection_index])
    if opt == OptKind.SHOP_BUY_RELIC:
        return Action(ActionType.ShopBuyRelic, [selection_index])
    if opt == OptKind.SHOP_BUY_POTION:
        return Action(ActionType.ShopBuyPotion, [selection_index])
    if opt == OptKind.SHOP_PURGE:
        return Action(ActionType.ShopPurge, [selection_index])
    if opt == OptKind.EVENT_OPTION:
        return Action(ActionType.EventOptionSelect, [selection_index])

    raise ValueError(f"Unhandled option kind {opt} for htp {htp}")
