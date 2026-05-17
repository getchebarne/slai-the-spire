"""
Mask generation for the actor-critic model.

Operates on `list[slai.GameState]` (not the old `EnvWrapper`s — wrapper
buffering for two-step targeting is gone, replaced by inline target
dispatch in `model.forward`).

Per-HeadTypePrimary outputs:
- `route[htp]`: int64 indices of samples in this primary group
- `primary_masks[htp]`: binary decision masks for decision primaries
- `selection_masks[htp]`: entity selection masks for all primaries

Plus auxiliary full-batch tensors (sample-indexed, used by the model
even for samples not in the corresponding primary group):
- `target_required`: (B, MAX_SIZE_HAND) bool — does each hand card need
  a target? Used for inline target dispatch in COMBAT_DEFAULT.
- `monster_alive_mask`: (B, MAX_MONSTERS) bool — alive monsters per
  sample. Used by inline HeadMonsterSelect.
- `retain_nums`: (B,) int — count of cards to retain this turn (only
  valid for samples in COMBAT_AWAIT_RETAIN).
"""

from dataclasses import dataclass

import numpy as np
import slai
import torch

from src.rl.action_space.route import get_route_primary
from src.rl.action_space.types import HeadTypePrimary
from src.rl.action_space.types import IS_DECISION_PRIMARY
from src.rl.action_space.types import NUM_PRIMARY_HEADS
from src.rl.action_space.types import PRIMARY_NUM_CHOICES
from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import card_identity_ids


def is_card_playable(card: slai.Card, energy_current: int) -> bool:
    """Single source of truth for whether a card can be played right now —
    affordability + per-card play restriction (slai's `Card.playable` flag
    captures Entangled, DrawPileEmpty, etc.)."""
    return card.cost <= energy_current and card.playable


# =============================================================================
# Constants (list-indexed by int(HeadTypePrimary))
# =============================================================================

_SELECTION_SIZES: list[int] = [0] * NUM_PRIMARY_HEADS
_SELECTION_SIZES[HeadTypePrimary.COMBAT_DEFAULT] = MAX_SIZE_HAND
_SELECTION_SIZES[HeadTypePrimary.CARD_REWARD] = MAX_SIZE_COMBAT_CARD_REWARD
_SELECTION_SIZES[HeadTypePrimary.REST_SITE] = MAX_SIZE_DECK
_SELECTION_SIZES[HeadTypePrimary.COMBAT_CARD_DISCARD] = MAX_SIZE_HAND
_SELECTION_SIZES[HeadTypePrimary.MAP_SELECT] = MAP_WIDTH
_SELECTION_SIZES[HeadTypePrimary.COMBAT_AWAIT_RETAIN] = MAX_SIZE_HAND
_SELECTION_SIZES[HeadTypePrimary.COMBAT_AWAIT_NIGHTMARE] = MAX_SIZE_HAND
_SELECTION_SIZES[HeadTypePrimary.COMBAT_AWAIT_SETUP] = MAX_SIZE_HAND
_SELECTION_SIZES[HeadTypePrimary.RELIC_REWARD] = MAX_RELIC_REWARDS

SELECTION_SIZES: tuple[int, ...] = tuple(_SELECTION_SIZES)


# =============================================================================
# MaskBatch
# =============================================================================


@dataclass
class MaskBatch:
    route: list[torch.Tensor]
    """sample indices per primary group. route[htp] → (N_group,) int64"""

    primary_masks: list[torch.Tensor]
    """decision masks. primary_masks[htp] → (N_group, num_choices) or empty"""

    selection_masks: list[torch.Tensor]
    """entity selection masks. selection_masks[htp] → (N_group, max_entities) or empty"""

    # ---- Per-sample auxiliary tensors (full batch B, sample-indexed) ----
    target_required: torch.Tensor | None
    """(B, MAX_SIZE_HAND) bool — does hand[i] require a monster target?
    Used by inline target dispatch in COMBAT_DEFAULT card-play. None on
    the PPO recompute path, which gates target replay on the recorded
    `target_index >= 0` and never reads this field."""

    monster_alive_mask: torch.Tensor
    """(B, MAX_MONSTERS) bool — alive monster slots per sample. Used by
    inline HeadMonsterSelect for COMBAT_DEFAULT targeting."""

    retain_nums: torch.Tensor
    """(B,) int — multi-pick count. Populated from `CombatAwaitRetain.num`
    AND `CombatAwaitDiscard.num` (slai's CardDiscard action requires
    exactly `num` indices in one shot, same shape as Retain).
    Zero outside those two phases. Field name kept for backwards-compat."""

    # ---- Per-pile group identity (for grouped sampling) ----
    # Two slots with the same group_id hold cards interchangeable for the
    # policy (same `_card_policy_features`). -1 marks padding/invalid slots.
    # Heads use these to dedup identical cards when sampling, so the policy
    # gradient is over CARD TYPES rather than slot positions.
    hand_group_ids: torch.Tensor
    """(B, MAX_SIZE_HAND) int — used by COMBAT_DEFAULT (card play),
    COMBAT_CARD_DISCARD, COMBAT_AWAIT_RETAIN/NIGHTMARE/SETUP."""

    deck_group_ids: torch.Tensor
    """(B, MAX_SIZE_DECK) int — used by REST_SITE (card upgrade)."""

    card_reward_group_ids: torch.Tensor
    """(B, MAX_SIZE_COMBAT_CARD_REWARD) int — used by CARD_REWARD."""


# =============================================================================
# Per-state primary mask functions (decision primaries only)
# =============================================================================


def _get_primary_mask_combat_default(state: slai.GameState) -> list[bool]:
    """[end_turn, play_card]"""
    has_playable = any(is_card_playable(card, state.energy.current) for card in state.hand)
    return [True, has_playable]


def _get_primary_mask_card_reward(state: slai.GameState) -> list[bool]:
    """[skip, select]"""
    can_select = len(state.deck) < MAX_SIZE_DECK and len(state.rewards_card) > 0
    return [True, can_select]


def _get_primary_mask_rest_site(state: slai.GameState) -> list[bool]:
    """[rest, upgrade]"""
    has_upgradable = any(not card.upgraded for card in state.deck)
    return [True, has_upgradable]


def _get_primary_mask_relic_reward(state: slai.GameState) -> list[bool]:
    """[skip, select]"""
    can_select = len(state.rewards_relic) > 0
    return [True, can_select]


_PRIMARY_MASK_FNS: list = [None] * NUM_PRIMARY_HEADS
_PRIMARY_MASK_FNS[HeadTypePrimary.COMBAT_DEFAULT] = _get_primary_mask_combat_default
_PRIMARY_MASK_FNS[HeadTypePrimary.CARD_REWARD] = _get_primary_mask_card_reward
_PRIMARY_MASK_FNS[HeadTypePrimary.REST_SITE] = _get_primary_mask_rest_site
_PRIMARY_MASK_FNS[HeadTypePrimary.RELIC_REWARD] = _get_primary_mask_relic_reward


# =============================================================================
# Per-state selection mask functions (all primaries)
# =============================================================================


def _get_selection_mask(htp: int, state: slai.GameState) -> list[bool]:
    """Get entity selection mask for a primary type (int-indexed)."""
    if htp == HeadTypePrimary.COMBAT_DEFAULT:
        mask = [False] * MAX_SIZE_HAND
        for idx, card in enumerate(state.hand[:MAX_SIZE_HAND]):
            mask[idx] = is_card_playable(card, state.energy.current)
        return mask

    if htp == HeadTypePrimary.COMBAT_CARD_DISCARD:
        mask = [False] * MAX_SIZE_HAND
        for idx in range(min(len(state.hand), MAX_SIZE_HAND)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.COMBAT_AWAIT_RETAIN:
        mask = [False] * MAX_SIZE_HAND
        for idx in range(min(len(state.hand), MAX_SIZE_HAND)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.COMBAT_AWAIT_NIGHTMARE:
        mask = [False] * MAX_SIZE_HAND
        for idx in range(min(len(state.hand), MAX_SIZE_HAND)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.COMBAT_AWAIT_SETUP:
        mask = [False] * MAX_SIZE_HAND
        for idx in range(min(len(state.hand), MAX_SIZE_HAND)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.CARD_REWARD:
        mask = [False] * MAX_SIZE_COMBAT_CARD_REWARD
        for idx in range(min(len(state.rewards_card), MAX_SIZE_COMBAT_CARD_REWARD)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.RELIC_REWARD:
        mask = [False] * MAX_RELIC_REWARDS
        for idx in range(min(len(state.rewards_relic), MAX_RELIC_REWARDS)):
            mask[idx] = True
        return mask

    if htp == HeadTypePrimary.REST_SITE:
        mask = [False] * MAX_SIZE_DECK
        for idx, card in enumerate(state.deck[:MAX_SIZE_DECK]):
            mask[idx] = not card.upgraded
        return mask

    if htp == HeadTypePrimary.MAP_SELECT:
        return _get_mask_map(state)

    raise ValueError(f"Unknown head type: {htp}")


def _get_mask_map(state: slai.GameState) -> list[bool]:
    """Mask for selectable map nodes (next-row columns reachable from cur)."""
    mask = [False] * MAP_WIDTH

    if not state.map.rooms:
        return mask

    if state.map.x_current is None and state.map.y_current is None:
        for x, room in enumerate(state.map.rooms[0][:MAP_WIDTH]):
            if room is not None:
                mask[x] = True
    else:
        y = state.map.y_current
        x = state.map.x_current

        if x is None or x < 0 or y is None or y >= len(state.map.rooms):
            return mask

        row = state.map.rooms[y]
        if x >= len(row):
            return mask

        current_room = row[x]
        if current_room is not None:
            for x_next in current_room.edges:
                if 0 <= x_next < MAP_WIDTH:
                    mask[x_next] = True

    return mask


# =============================================================================
# Per-sample auxiliary tensors
# =============================================================================


def _get_target_required(state: slai.GameState) -> list[bool]:
    """Per-hand-slot: True if playing this card requires a monster target."""
    mask = [False] * MAX_SIZE_HAND
    for idx, card in enumerate(state.hand[:MAX_SIZE_HAND]):
        mask[idx] = card.requires_target
    return mask


def _get_monster_alive_mask(state: slai.GameState) -> list[bool]:
    """Per-monster-slot: True if monster at this slot is alive (i.e. exists
    in the alive-only `state.monsters` list)."""
    mask = [False] * MAX_MONSTERS
    for idx in range(min(len(state.monsters), MAX_MONSTERS)):
        mask[idx] = True
    return mask


def _get_retain_num(state: slai.GameState) -> int:
    """Multi-pick count for samples in CombatAwaitRetain or CombatAwaitDiscard.
    Both phases use the same multi-pick machinery — slai's CardDiscard
    expects `num` indices in one shot, identical shape to CardRetain."""
    if isinstance(state.phase, slai.Phase.CombatAwaitRetain):
        return int(state.phase.num)
    if isinstance(state.phase, slai.Phase.CombatAwaitDiscard):
        return int(state.phase.num)
    return 0


# =============================================================================
# Public API
# =============================================================================


def get_mask_batch(
    states: list[slai.GameState],
    device: torch.device,
) -> MaskBatch:
    """
    Build MaskBatch for a list of slai.GameState.

    Routes states by phase, then generates primary, selection, and
    auxiliary masks.
    """
    B = len(states)
    route_lists = get_route_primary(states)

    route: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore
    primary_masks: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore
    selection_masks: list[torch.Tensor] = [None] * NUM_PRIMARY_HEADS  # type: ignore

    for htp in range(NUM_PRIMARY_HEADS):
        indices = route_lists[htp]
        route[htp] = torch.tensor(indices, dtype=torch.long, device=device)

        n = len(indices)
        sel_size = SELECTION_SIZES[htp]

        if n == 0:
            if IS_DECISION_PRIMARY[htp]:
                primary_masks[htp] = torch.zeros(
                    0, PRIMARY_NUM_CHOICES[htp], dtype=torch.bool, device=device
                )
            else:
                primary_masks[htp] = torch.empty(0, dtype=torch.bool, device=device)
            selection_masks[htp] = torch.zeros(0, sel_size, dtype=torch.bool, device=device)
            continue

        group_states = [states[i] for i in indices]

        # Primary masks (decision primaries only)
        if IS_DECISION_PRIMARY[htp]:
            num_choices = PRIMARY_NUM_CHOICES[htp]
            primary_fn = _PRIMARY_MASK_FNS[htp]
            primary_np = np.zeros((n, num_choices), dtype=bool)
            for b, state in enumerate(group_states):
                primary_np[b] = primary_fn(state)
            primary_masks[htp] = torch.from_numpy(primary_np).to(device)
        else:
            primary_masks[htp] = torch.empty(0, dtype=torch.bool, device=device)

        # Selection masks (all groups)
        sel_np = np.zeros((n, sel_size), dtype=bool)
        for b, state in enumerate(group_states):
            sel_np[b] = _get_selection_mask(htp, state)
        selection_masks[htp] = torch.from_numpy(sel_np).to(device)

    # Per-sample auxiliary tensors (full batch indexing)
    target_req_np = np.zeros((B, MAX_SIZE_HAND), dtype=bool)
    monster_alive_np = np.zeros((B, MAX_MONSTERS), dtype=bool)
    retain_nums_np = np.zeros(B, dtype=np.int64)
    hand_gids_np = np.full((B, MAX_SIZE_HAND), -1, dtype=np.int64)
    deck_gids_np = np.full((B, MAX_SIZE_DECK), -1, dtype=np.int64)
    reward_gids_np = np.full((B, MAX_SIZE_COMBAT_CARD_REWARD), -1, dtype=np.int64)
    for i, state in enumerate(states):
        target_req_np[i] = _get_target_required(state)
        monster_alive_np[i] = _get_monster_alive_mask(state)
        retain_nums_np[i] = _get_retain_num(state)

        hand_ids = card_identity_ids(state.hand[:MAX_SIZE_HAND])
        hand_gids_np[i, : len(hand_ids)] = hand_ids
        deck_ids = card_identity_ids(state.deck[:MAX_SIZE_DECK])
        deck_gids_np[i, : len(deck_ids)] = deck_ids
        reward_ids = card_identity_ids(state.rewards_card[:MAX_SIZE_COMBAT_CARD_REWARD])
        reward_gids_np[i, : len(reward_ids)] = reward_ids

    target_required = torch.from_numpy(target_req_np).to(device)
    monster_alive_mask = torch.from_numpy(monster_alive_np).to(device)
    retain_nums = torch.from_numpy(retain_nums_np).to(device)
    hand_group_ids = torch.from_numpy(hand_gids_np).to(device)
    deck_group_ids = torch.from_numpy(deck_gids_np).to(device)
    card_reward_group_ids = torch.from_numpy(reward_gids_np).to(device)

    return MaskBatch(
        route=route,
        primary_masks=primary_masks,
        selection_masks=selection_masks,
        target_required=target_required,
        monster_alive_mask=monster_alive_mask,
        retain_nums=retain_nums,
        hand_group_ids=hand_group_ids,
        deck_group_ids=deck_group_ids,
        card_reward_group_ids=card_reward_group_ids,
    )
