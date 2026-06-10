"""Action masks, derived directly from the engine's `get_legal_actions()`.

The engine emits the authoritative legal-action list for each state; we fold it into
three masks (no separate route module, no `state.pending`, no `operation` — a halt is
just a state where one action type is legal, which the L1 mask already encodes):
  - `mask_action_type` (B, NUM_ACTION_TYPES): L1 — which action KINDs are legal. EVERY
    legal action type is folded in (incl. pending picks); a halt has exactly one bit set.
  - `mask_action_idx` (TensorDict {str(int(ActionType)): (B, pool_size) bool}): L2 — which
    entities are legal for THAT action type, identity-deduped. Keyed per action type (not Pool)
    because legality is per-type (PotionUse vs PotionDiscard share the POTIONS pool but
    differ — a combat-only potion outside combat is discardable but not usable). Absent
    for terminal types (no selection).
  - `mask_target_card` / `mask_target_potion` (B, N, MAX_MONSTERS): L3 — legal
    (entity, monster) pairs, from the engine's per-monster CardPlay/PotionUse variants.
    `requires_target(entity)` is just that row's `.any()` over monsters.
"""

import warnings

import numpy as np
import slai
import torch
from tensordict import TensorDict

from src.rl.types import AT_POOL
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import POOL_SIZE
from src.rl.types import Pool
from src.rl.types import SELECTING_ACTION_TYPES
from src.rl.types import TMask
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_CARDS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import card_identity_key


def is_card_playable(card: slai.Card, energy_current: int) -> bool:
    """Affordability + per-card play restriction (kept for test_agent display)."""
    return card.cost <= energy_current and card.playable


def card_identity_ids(cards: list[slai.Card]) -> list[int]:
    """Contiguous group ids (from 0) over a pile; cards with the same identity
    key share an id (selection dedup). One id per card; the caller pads
    truncated/empty slots with -1."""
    seen: dict[tuple, int] = {}
    return [seen.setdefault(card_identity_key(card), len(seen)) for card in cards]


def _pile_ids(state: slai.GameState, pool: int) -> list[int]:
    """Identity group-ids for the pool's dedup source pile; [] if the pool doesn't dedup
    (only card piles dedup — copies of the same card collapse to one selectable slot)."""
    if pool == Pool.HAND:
        return card_identity_ids(state.hand[:MAX_SIZE_HAND])
    if pool == Pool.DECK:
        return card_identity_ids(state.deck[:MAX_SIZE_DECK])
    if pool == Pool.REWARD_CARDS:
        cards = state.reward.cards if state.reward is not None else []
        return card_identity_ids(cards[:MAX_SIZE_COMBAT_CARD_REWARD])
    if pool == Pool.DISCOVER:
        return card_identity_ids(state.discover[:MAX_SIZE_DISCOVER])
    if pool == Pool.SHOP_CARDS:
        cards = state.shop.cards if state.shop is not None else []
        return card_identity_ids(cards[:MAX_SHOP_CARDS])
    return []


_POOL_HAND = int(Pool.HAND)
_POOL_POTIONS = int(Pool.POTIONS)

# Action types we've already warned about dropping (selection index beyond the
# encoder's pool cap) — truncation must never be silent.
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    legal_actions_batch: list[list],
    device: torch.device,
) -> TMask:
    """Build a TMask from each state's engine-emitted legal actions. Every legal
    action type sets its L1 bit (a halt ends up with exactly one); each action type's
    legal entities go in its own L2 mask (legality is per type, deduped per pool)."""
    B = len(states)
    action_type_np = np.zeros((B, NUM_ACTION_TYPES), dtype=bool)
    sel_np = {at: np.zeros((B, POOL_SIZE[AT_POOL[at]]), dtype=bool) for at in SELECTING_ACTION_TYPES}
    card_tgt_np = np.zeros((B, MAX_SIZE_HAND, MAX_MONSTERS), dtype=bool)
    potion_tgt_np = np.zeros((B, MAX_POTION_SLOTS, MAX_MONSTERS), dtype=bool)

    for i, state in enumerate(states):
        for action in legal_actions_batch[i]:
            at = int(action.action_type)
            idxs = action.idxs
            pool = AT_POOL[at]
            if pool >= 0 and idxs:
                if idxs[0] >= POOL_SIZE[pool]:
                    # Beyond the encoder cap: unselectable, so don't legalize its
                    # L1 kind off it either (a legal kind with an all-False L2 mask
                    # would NaN the selection Categorical).
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(
                            f"Dropped legal action: type={at} idx={idxs[0]} exceeds "
                            f"pool cap {POOL_SIZE[pool]} (see constants.MAX_SIZE_*)"
                        )
                    continue
                sel_np[at][i, idxs[0]] = True  # per ACTION TYPE — legality differs per type
                if len(idxs) == 2 and idxs[1] < MAX_MONSTERS:
                    if pool == _POOL_HAND:
                        card_tgt_np[i, idxs[0], idxs[1]] = True
                    elif pool == _POOL_POTIONS:
                        potion_tgt_np[i, idxs[0], idxs[1]] = True
            action_type_np[i, at] = True  # L1: every legal kind the masks can express
        if not action_type_np[i].any():
            # Every legal action was beyond a pool cap — unrepresentable state;
            # fail loudly instead of sampling NaNs.
            raise RuntimeError(
                f"All legal actions exceed encoder pool caps on screen {state.screen}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[i]]}"
            )

        # Dedup each action type's selection mask via its pool's identity pile: keep
        # only the first occurrence of each card identity (copies collapse to one).
        # Non-deduping pools return [] from _pile_ids, so the empty check skips them.
        for at in SELECTING_ACTION_TYPES:
            pool = AT_POOL[at]
            row = sel_np[at][i]
            if not row.any():
                continue
            ids = _pile_ids(state, pool)
            if not ids:
                continue
            seen: set = set()
            for slot in range(min(len(ids), POOL_SIZE[pool])):
                if not row[slot]:
                    continue
                if ids[slot] in seen:
                    row[slot] = False
                else:
                    seen.add(ids[slot])

    return TMask(
        mask_action_type=torch.from_numpy(action_type_np),
        mask_action_idx=TensorDict(
            {str(at): torch.from_numpy(sel_np[at]) for at in SELECTING_ACTION_TYPES}, batch_size=[B]
        ),
        mask_target_card=torch.from_numpy(card_tgt_np),
        mask_target_potion=torch.from_numpy(potion_tgt_np),
        batch_size=[B],
    ).to(device)
