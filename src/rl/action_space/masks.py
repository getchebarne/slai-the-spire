"""Action masks, derived directly from the engine's `get_legal_actions()`.

Rather than re-implement the engine's per-screen legality rules in Python (which
drifts — e.g. event-consumed and shop-purge-per-visit aren't derivable from the
snapshot alone), we build masks from the authoritative legal-action list the
engine emits for each state. This guarantees the policy can only choose legal
actions and keeps routing/masking exactly in step with `recompute_legal_actions`.

Each legal `Action` carries `action_type` + `idxs`. We fold them into:
  - `option_masks[htp]`: which OptKinds are legal for each screen-head sample
  - `sel_masks[selkey]`: which entity indices are legal for each selection head
  - `target_required_*`: whether a CardPlay/PotionUse needs a monster target
  - `multipick_mask` / `pick_nums`: hand mask + count for discard/retain
  - `monster_alive_mask`: alive monster slots (the legal target set)
"""

from dataclasses import dataclass

import numpy as np
import slai
import torch

from src.rl.action_space.route import get_route_primary
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import NUM_SEL_KEYS
from src.rl.action_space.types import PEND_SINGLE_SELKEY
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.action_space.types import SCREEN_OPTION_KINDS
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import OptKind
from src.rl.action_space.types import SelKey
from src.rl.action_space.types import opt_sel_key
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import card_identity_ids


def is_card_playable(card: slai.Card, energy_current: int) -> bool:
    """Affordability + per-card play restriction (kept for test_agent display)."""
    return card.cost <= energy_current and card.playable


# =============================================================================
# Per-SelKey pool size + group-id source pile
# =============================================================================

SEL_POOL_SIZE: list[int] = [0] * NUM_SEL_KEYS
SEL_POOL_SIZE[SelKey.CARD_PLAY] = MAX_SIZE_HAND
SEL_POOL_SIZE[SelKey.POTION_USE] = MAX_POTION_SLOTS
SEL_POOL_SIZE[SelKey.POTION_DISCARD] = MAX_POTION_SLOTS
SEL_POOL_SIZE[SelKey.ROOM_SELECT] = MAP_WIDTH
SEL_POOL_SIZE[SelKey.REST_UPGRADE] = MAX_SIZE_DECK
SEL_POOL_SIZE[SelKey.REWARD_CARD] = MAX_SIZE_COMBAT_CARD_REWARD
SEL_POOL_SIZE[SelKey.SHOP_CARD] = MAX_SHOP_CARDS
SEL_POOL_SIZE[SelKey.SHOP_RELIC] = MAX_SHOP_RELICS
SEL_POOL_SIZE[SelKey.SHOP_POTION] = MAX_SHOP_POTIONS
SEL_POOL_SIZE[SelKey.SHOP_PURGE] = MAX_SIZE_DECK
SEL_POOL_SIZE[SelKey.EVENT_OPTION] = MAX_EVENT_OPTIONS
SEL_POOL_SIZE[SelKey.PEND_SETUP] = MAX_SIZE_HAND
SEL_POOL_SIZE[SelKey.PEND_NIGHTMARE] = MAX_SIZE_HAND
SEL_POOL_SIZE[SelKey.PEND_DISCOVER] = MAX_SIZE_DISCOVER
SEL_POOL_SIZE[SelKey.PEND_PURGE] = MAX_SIZE_DECK
SEL_POOL_SIZE[SelKey.PEND_UPGRADE] = MAX_SIZE_DECK
SEL_POOL_SIZE[SelKey.PEND_DUPLICATE] = MAX_SIZE_DECK
SEL_POOL_SIZE[SelKey.PEND_TRANSFORM] = MAX_SIZE_DECK

# Which card pile (for grouped sampling dedup) each SelKey draws from; None = no dedup.
_PILE_HAND, _PILE_DECK, _PILE_REWARD, _PILE_DISCOVER, _PILE_SHOP_CARD = (
    "hand",
    "deck",
    "reward",
    "discover",
    "shop_card",
)
SEL_GROUP_PILE: list = [None] * NUM_SEL_KEYS
SEL_GROUP_PILE[SelKey.CARD_PLAY] = _PILE_HAND
SEL_GROUP_PILE[SelKey.PEND_SETUP] = _PILE_HAND
SEL_GROUP_PILE[SelKey.PEND_NIGHTMARE] = _PILE_HAND
SEL_GROUP_PILE[SelKey.REST_UPGRADE] = _PILE_DECK
SEL_GROUP_PILE[SelKey.SHOP_PURGE] = _PILE_DECK
SEL_GROUP_PILE[SelKey.PEND_PURGE] = _PILE_DECK
SEL_GROUP_PILE[SelKey.PEND_UPGRADE] = _PILE_DECK
SEL_GROUP_PILE[SelKey.PEND_DUPLICATE] = _PILE_DECK
SEL_GROUP_PILE[SelKey.PEND_TRANSFORM] = _PILE_DECK
SEL_GROUP_PILE[SelKey.REWARD_CARD] = _PILE_REWARD
SEL_GROUP_PILE[SelKey.PEND_DISCOVER] = _PILE_DISCOVER
SEL_GROUP_PILE[SelKey.SHOP_CARD] = _PILE_SHOP_CARD


# ActionType (screen-context) → OptKind. CardUpgrade at a screen is the rest-site
# upgrade (pending deck-upgrade is routed to a PEND head, handled separately).
_AT = slai.ActionType
_SCREEN_AT_TO_OPT = {
    int(_AT.TurnEnd): OptKind.TURN_END,
    int(_AT.CardPlay): OptKind.CARD_PLAY,
    int(_AT.PotionUse): OptKind.USE_POTION,
    int(_AT.PotionDiscard): OptKind.DISCARD_POTION,
    int(_AT.RoomSelect): OptKind.ROOM_SELECT,
    int(_AT.Rest): OptKind.REST,
    int(_AT.CardUpgrade): OptKind.REST_UPGRADE,
    int(_AT.RoomExit): OptKind.ROOM_EXIT,
    int(_AT.ChestOpen): OptKind.CHEST_OPEN,
    int(_AT.RewardTakeCard): OptKind.REWARD_TAKE_CARD,
    int(_AT.RewardTakeRelic): OptKind.REWARD_TAKE_RELIC,
    int(_AT.RewardTakePotion): OptKind.REWARD_TAKE_POTION,
    int(_AT.RewardTakeGold): OptKind.REWARD_TAKE_GOLD,
    int(_AT.ShopBuyCard): OptKind.SHOP_BUY_CARD,
    int(_AT.ShopBuyRelic): OptKind.SHOP_BUY_RELIC,
    int(_AT.ShopBuyPotion): OptKind.SHOP_BUY_POTION,
    int(_AT.ShopPurge): OptKind.SHOP_PURGE,
    int(_AT.EventOptionSelect): OptKind.EVENT_OPTION,
}

# Per-screen-htp: OptKind -> position in the option categorical.
_OPT_POS: dict = {
    htp: {opt: j for j, opt in enumerate(opts)} for htp, opts in SCREEN_OPTION_KINDS.items()
}


@dataclass
class MaskBatch:
    route: list  # per htp: (N_htp,) int64 sample indices
    option_masks: list  # per htp: (N_htp, K_htp) bool (empty for non-screen heads)
    sel_masks: list  # per SelKey: (B, pool_size) bool
    sel_group_ids: list  # per SelKey: (B, pool_size) int64 or None
    monster_alive_mask: torch.Tensor  # (B, MAX_MONSTERS) bool
    target_required_hand: torch.Tensor  # (B, MAX_SIZE_HAND) bool
    target_required_potion: torch.Tensor  # (B, MAX_POTION_SLOTS) bool
    multipick_mask: torch.Tensor  # (B, MAX_SIZE_HAND) bool — hand avail for discard/retain
    multipick_group_ids: torch.Tensor  # (B, MAX_SIZE_HAND) int64
    pick_nums: torch.Tensor  # (B,) int64 — discard/retain count


def _pile_ids(state: slai.GameState, pile: str) -> list[int]:
    if pile == _PILE_HAND:
        return card_identity_ids(state.hand[:MAX_SIZE_HAND])
    if pile == _PILE_DECK:
        return card_identity_ids(state.deck[:MAX_SIZE_DECK])
    if pile == _PILE_REWARD:
        cards = state.reward.cards if state.reward is not None else []
        return card_identity_ids(cards[:MAX_SIZE_COMBAT_CARD_REWARD])
    if pile == _PILE_DISCOVER:
        return card_identity_ids(state.discover[:MAX_SIZE_DISCOVER])
    if pile == _PILE_SHOP_CARD:
        cards = state.shop.cards if state.shop is not None else []
        return card_identity_ids(cards[:MAX_SHOP_CARDS])
    return []


def get_mask_batch(
    states: list[slai.GameState],
    legal_actions_batch: list[list],
    device: torch.device,
) -> MaskBatch:
    """Build a MaskBatch from each state's engine-emitted legal actions."""
    B = len(states)
    route_lists = get_route_primary(states)

    # Per-SelKey full-batch masks + group ids
    sel_np = [np.zeros((B, SEL_POOL_SIZE[k]), dtype=bool) for k in range(NUM_SEL_KEYS)]
    gid_np = [
        np.full((B, SEL_POOL_SIZE[k]), -1, dtype=np.int64)
        if SEL_GROUP_PILE[k] is not None
        else None
        for k in range(NUM_SEL_KEYS)
    ]
    monster_alive_np = np.zeros((B, MAX_MONSTERS), dtype=bool)
    tgt_hand_np = np.zeros((B, MAX_SIZE_HAND), dtype=bool)
    tgt_potion_np = np.zeros((B, MAX_POTION_SLOTS), dtype=bool)
    multipick_np = np.zeros((B, MAX_SIZE_HAND), dtype=bool)
    multipick_gid_np = np.full((B, MAX_SIZE_HAND), -1, dtype=np.int64)
    pick_nums_np = np.zeros(B, dtype=np.int64)

    # Per-sample option-mask rows (screen heads only)
    option_rows: dict = {}

    # htp lookup per sample
    htp_of = [None] * B
    for htp in range(NUM_PRIMARY_HEADS):
        for i in route_lists[htp]:
            htp_of[i] = htp

    for i, state in enumerate(states):
        htp = htp_of[i]
        legal = legal_actions_batch[i]

        n_alive = min(len(state.monsters), MAX_MONSTERS)
        monster_alive_np[i, :n_alive] = True

        # Fill group ids for every pile (cheap; only routed slots are read by heads)
        for k in range(NUM_SEL_KEYS):
            pile = SEL_GROUP_PILE[k]
            if pile is None:
                continue
            ids = _pile_ids(state, pile)
            n = min(len(ids), SEL_POOL_SIZE[k])
            if n:
                gid_np[k][i, :n] = ids[:n]

        if htp in SCREEN_OPTION_KINDS:
            opts = SCREEN_OPTION_KINDS[htp]
            row = np.zeros(len(opts), dtype=bool)
            pos = _OPT_POS[htp]
            for action in legal:
                opt = _SCREEN_AT_TO_OPT.get(int(action.action_type))
                if opt is None or opt not in pos:
                    continue
                row[pos[opt]] = True
                selkey = opt_sel_key(opt)
                if selkey is None:
                    continue
                idxs = action.idxs
                idx0 = idxs[0]
                if idx0 < SEL_POOL_SIZE[selkey]:
                    sel_np[selkey][i, idx0] = True
                if opt == OptKind.CARD_PLAY and len(idxs) == 2 and idx0 < MAX_SIZE_HAND:
                    tgt_hand_np[i, idx0] = True
                if opt == OptKind.USE_POTION and len(idxs) == 2 and idx0 < MAX_POTION_SLOTS:
                    tgt_potion_np[i, idx0] = True
            option_rows[i] = row

        elif htp in (HeadTypePrimary.PEND_DISCARD, HeadTypePrimary.PEND_RETAIN):
            hand_ids = card_identity_ids(state.hand[:MAX_SIZE_HAND])
            n = min(len(hand_ids), MAX_SIZE_HAND)
            multipick_gid_np[i, :n] = hand_ids[:n]
            for action in legal:
                for idx in action.idxs:
                    if idx < MAX_SIZE_HAND:
                        multipick_np[i, idx] = True
            if legal:
                pick_nums_np[i] = len(legal[0].idxs)

        else:
            # Pending single-select
            selkey = PEND_SINGLE_SELKEY[htp]
            for action in legal:
                idx0 = action.idxs[0]
                if idx0 < SEL_POOL_SIZE[selkey]:
                    sel_np[selkey][i, idx0] = True

    # Assemble per-htp tensors
    route = [
        torch.tensor(route_lists[htp], dtype=torch.long, device=device)
        for htp in range(NUM_PRIMARY_HEADS)
    ]
    option_masks = []
    for htp in range(NUM_PRIMARY_HEADS):
        idxs = route_lists[htp]
        k = PRIMARY_NUM_CHOICES[htp]
        if k == 0 or not idxs:
            option_masks.append(torch.zeros(len(idxs), k, dtype=torch.bool, device=device))
            continue
        rows = np.stack([option_rows[i] for i in idxs], axis=0)
        option_masks.append(torch.from_numpy(rows).to(device))

    sel_masks = [torch.from_numpy(sel_np[k]).to(device) for k in range(NUM_SEL_KEYS)]
    sel_group_ids = [
        torch.from_numpy(gid_np[k]).to(device) if gid_np[k] is not None else None
        for k in range(NUM_SEL_KEYS)
    ]

    return MaskBatch(
        route=route,
        option_masks=option_masks,
        sel_masks=sel_masks,
        sel_group_ids=sel_group_ids,
        monster_alive_mask=torch.from_numpy(monster_alive_np).to(device),
        target_required_hand=torch.from_numpy(tgt_hand_np).to(device),
        target_required_potion=torch.from_numpy(tgt_potion_np).to(device),
        multipick_mask=torch.from_numpy(multipick_np).to(device),
        multipick_group_ids=torch.from_numpy(multipick_gid_np).to(device),
        pick_nums=torch.from_numpy(pick_nums_np).to(device),
    )
