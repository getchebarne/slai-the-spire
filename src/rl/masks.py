import warnings

import numpy as np
import slai
import torch
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
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import TMask


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


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup)."""
    seen: dict[tuple, int] = {}
    return [seen.setdefault(card.identity_hash, len(seen)) for card in cards]


def _pile_ids(state: slai.GameState, target_kind: "SliceKind | None") -> list[int]:
    """Identity group-ids for a card target's dedup pile; [] if the target doesn't dedup."""
    if target_kind not in SLICE_KIND_CARDS:
        return []
    return card_identity_ids(_get_card_entities(target_kind, state)[: BY_KIND[target_kind].size])


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
        for action in legal_actions_batch[i]:
            at = int(action.action_type)
            idxs = action.idxs
            target_kind = ACTION_TYPE_POOL.get(action.action_type)
            if target_kind is not None and idxs:
                if idxs[0] >= ACTION_TYPE_SIZE[at]:
                    # Beyond cap: unselectable — drop its action-type bit (all-False l1 mask NaNs)
                    if at not in _WARNED_DROPPED:
                        _WARNED_DROPPED.add(at)
                        warnings.warn(
                            f"Dropped legal action: type={at} idx={idxs[0]} exceeds "
                            f"pool cap {ACTION_TYPE_SIZE[at]} (see constants.MAX_SIZE_*)"
                        )
                    continue
                np_l1[at][i, idxs[0]] = True  # per ACTION TYPE — legality differs per type
                if len(idxs) == 2 and idxs[1] < MAX_MONSTERS:
                    if target_kind in SLICE_KIND_CARDS:
                        np_card_l2[i, idxs[0], idxs[1]] = True
                    elif target_kind in SLICE_KIND_POTIONS:
                        np_potion_l2[i, idxs[0], idxs[1]] = True
            np_action_type[i, at] = True  # ActionType: every legal kind the masks can express
        if not np_action_type[i].any():
            # Every legal action exceeded a pool cap — unrepresentable; fail loudly not NaN.
            raise RuntimeError(
                f"All legal actions exceed encoder pool caps on screen {state.screen}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[i]]}"
            )

        # Dedup each type's l1 mask by identity pile (copies collapse); non-dedup pools return [].
        for action_type, target_kind in ACTION_TYPE_POOL.items():
            at = int(action_type)
            np_row = np_l1[at][i]
            if not np_row.any():
                continue
            ids = _pile_ids(state, target_kind)
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
