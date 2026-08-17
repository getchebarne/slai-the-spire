import warnings

import numpy as np
import slai
import torch
from slai import ActionType
from slai import Card
from tensordict import TensorDict

from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SIZE_HAND
from src.rl.encoding.card import SLICE_CARDS
from src.rl.encoding.card import SLICE_KIND_CARDS
from src.rl.encoding.card import _get_card_entities
from src.rl.encoding.event import SLICE_EVENTS
from src.rl.encoding.map_ import SLICE_ROOMS
from src.rl.encoding.potion import SLICE_KIND_POTIONS
from src.rl.encoding.potion import SLICE_POTIONS
from src.rl.encoding.relic import SLICE_RELICS
from src.rl.encoding.screen import derive_focus
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import TMask
from src.rl.utils import reward_card_prefix
from src.rl.utils import reward_slot


# Target-slice sizes; built here (masks is the only consumer).
BY_KIND: dict[SliceKind, Slice] = {
    slice_.kind: slice_
    for table in (SLICE_CARDS, SLICE_RELICS, SLICE_POTIONS, SLICE_EVENTS, SLICE_ROOMS)
    for slice_ in table
}
# Per-ActionType l1 mask width (the pool's cap), for the selecting actions.
ACTION_TYPE_SIZE: dict[int, int] = {
    int(action_type): BY_KIND[pool].size for action_type, pool in ACTION_TYPE_POOL.items()
}
# Action types with an l1 selection.
SELECTING_ACTION_TYPES: list[int] = [int(action_type) for action_type in ACTION_TYPE_POOL]

# The reward-card actions: their L1 slot is a position in the bundle-flattened
# CARD_REWARD slice (translation via utils.reward_slot / reward_card_prefix).
_AT_REWARD_CARD = frozenset((int(ActionType.RewardTakeCard), int(ActionType.RewardSingingBowl)))
# The only action types whose second engine idx is a monster target (L3).
_AT_MONSTER_TARGETED = frozenset((int(ActionType.CardPlay), int(ActionType.PotionUse)))


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup)."""
    seen: dict[Card, int] = {}
    return [seen.setdefault(card, len(seen)) for card in cards]


def _pile_ids(state: slai.GameState, combat, target_kind: "SliceKind | None") -> list[int]:
    """Identity group-ids for a card target's dedup pile; [] if the target doesn't dedup.

    `combat` is the caller's hoisted `state.combat` (FFI accesses deep-clone)."""
    if target_kind not in SLICE_KIND_CARDS:
        return []
    if target_kind is SliceKind.CARD_REWARD:
        # No dedup across bundles: the same card offered in two bundles is two
        # DIFFERENT actions (taking one consumes its whole bundle).
        return []
    return card_identity_ids(
        _get_card_entities(target_kind, state, combat)[: BY_KIND[target_kind].size]
    )


# Action types already warned about dropped selections — truncation is never silent.
_WARNED_DROPPED: set[int] = set()


def build_masks(
    states: list[slai.GameState],
    legal_actions_batch: list[list],
    device: torch.device,
) -> TMask:
    """Build a TMask from each state's legal actions (action-type bits + per-type l1 masks)."""
    B = len(states)
    np_action_type = np.zeros((B, NUM_ACTION_TYPES), dtype=bool)
    np_l1 = {at: np.zeros((B, ACTION_TYPE_SIZE[at]), dtype=bool) for at in SELECTING_ACTION_TYPES}
    np_card_l2 = np.zeros((B, MAX_SIZE_HAND, MAX_MONSTERS), dtype=bool)
    np_potion_l2 = np.zeros((B, MAX_POTION_SLOTS, MAX_MONSTERS), dtype=bool)

    for i, state in enumerate(states):
        prefix = reward_card_prefix(state)  # cheap outside reward states ([0])
        combat = state.combat  # hoisted once: FFI context accesses deep-clone
        for action in legal_actions_batch[i]:
            at = int(action.action_type)
            idxs = action.idxs
            target_kind = ACTION_TYPE_POOL.get(action.action_type)
            if target_kind is not None and idxs:
                # L1 slot: bundle-flattened for the reward-card actions, the raw
                # engine index everywhere else.
                if at in _AT_REWARD_CARD:
                    slot = reward_slot(prefix, action.action_type, idxs)
                else:
                    slot = idxs[0]
                if slot >= ACTION_TYPE_SIZE[at]:
                    # Beyond cap: unselectable — drop its action-type bit (all-False l1 mask NaNs)
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(
                            f"Dropped legal action: type={at} slot={slot} exceeds "
                            f"pool cap {ACTION_TYPE_SIZE[at]} (see constants.MAX_SIZE_*)"
                        )
                    continue
                np_l1[at][i, slot] = True  # per ACTION TYPE — legality differs per type
                # L3: only CardPlay/PotionUse carry a monster target as idxs[1]
                # (RewardTakeCard's idxs[1] is a card-within-bundle, folded above).
                if at in _AT_MONSTER_TARGETED and len(idxs) == 2 and idxs[1] < MAX_MONSTERS:
                    if target_kind in SLICE_KIND_CARDS:
                        np_card_l2[i, slot, idxs[1]] = True
                    elif target_kind in SLICE_KIND_POTIONS:
                        np_potion_l2[i, slot, idxs[1]] = True
            np_action_type[i, at] = True  # ActionType: every legal kind the masks can express
        if not np_action_type[i].any():
            # Every legal action exceeded a pool cap — unrepresentable; fail loudly not NaN.
            raise RuntimeError(
                f"All legal actions exceed encoder pool caps on focus "
                f"{derive_focus(state).name}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[i]]}"
            )

        # Dedup each type's l1 mask by identity pile (copies collapse); non-dedup pools return [].
        for action_type, target_kind in ACTION_TYPE_POOL.items():
            at = int(action_type)
            np_row = np_l1[at][i]
            if not np_row.any():
                continue
            ids = _pile_ids(state, combat, target_kind)
            if not ids:
                continue
            seen: set = set()
            for slot in range(min(len(ids), ACTION_TYPE_SIZE[at])):
                if not np_row[slot]:
                    continue
                if ids[slot] in seen:
                    np_row[slot] = False
                else:
                    seen.add(ids[slot])

    return TMask(
        mask_action_type=torch.from_numpy(np_action_type),
        mask_action_idx=TensorDict(
            {str(at): torch.from_numpy(np_l1[at]) for at in SELECTING_ACTION_TYPES},
            batch_size=[B],
        ),
        mask_target_card=torch.from_numpy(np_card_l2),
        mask_target_potion=torch.from_numpy(np_potion_l2),
        batch_size=[B],
    ).to(device)
