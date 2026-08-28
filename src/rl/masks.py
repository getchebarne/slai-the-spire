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
from src.rl.encoding.card import SLICE_KIND_CARDS
from src.rl.encoding.card import _get_card_entities
from src.rl.encoding.card import card_key
from src.rl.encoding.effect import effect_key
from src.rl.encoding.focus import derive_focus
from src.rl.encoding.layout import SLICE_ALL
from src.rl.encoding.potion import SLICE_KIND_POTIONS
from src.rl.types import ACTION_TYPE_POOL
from src.rl.types import NUM_ACTION_TYPES
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import TMask
from src.rl.utils import pending_virtual_int
from src.rl.utils import reward_slot


# Target-slice sizes; built here (masks is the only consumer)
_BY_KIND: dict[SliceKind, Slice] = {slice_.kind: slice_ for slice_ in SLICE_ALL}
# Per-policy-type l1 mask width (the pool's cap), for the selecting actions
ACTION_TYPE_SIZE: dict[int, int] = {
    at: sum(_BY_KIND[kind].size for kind in kinds) for at, kinds in ACTION_TYPE_POOL.items()
}
# Policy action types with an l1 selection (engine + virtual pending)
SELECTING_ACTION_TYPES: list[int] = list(ACTION_TYPE_POOL)

# The reward-card actions: their L1 slot is a position in the concatenated
# per-bundle reward slices (translation via utils.reward_slot)
_AT_REWARD_CARD = frozenset((int(ActionType.RewardTakeCard), int(ActionType.RewardSingingBowl)))
# The only action types whose second engine idx is a monster target (L3)
_AT_MONSTER_TARGETED = frozenset((int(ActionType.CardPlay), int(ActionType.PotionUse)))


def card_identity_ids(cards: list[Card]) -> list[int]:
    """Contiguous group ids over a pile; same-identity cards share an id (selection dedup).

    Groups by (card_key, effect_key) — the card row plus the effect rows, the
    card's full model-visible identity. Two same-name cards can carry different
    effects (Glass Knife decay, Ritual Dagger bumps) and are NOT interchangeable
    picks. Not keyed on the Card object: the engine's __hash__ covers the arena
    `id`, which would make every card distinct and the dedup a no-op."""
    seen: dict[tuple, int] = {}
    return [
        seen.setdefault((card_key(card), effect_key(card.effects)), len(seen)) for card in cards
    ]


def _pile_ids(
    state: slai.GameState,
    combat: "slai.Combat | None",
    target_kinds: "tuple[SliceKind, ...] | None",
) -> list[int]:
    """Identity group-ids for a card target's dedup pile; [] if the target doesn't dedup.

    Multi-slice pools are the reward bundles and never dedup: the same card
    offered in two bundles is two DIFFERENT actions (taking one consumes its
    whole bundle). `combat` is the caller's hoisted `state.combat`."""
    if target_kinds is None or len(target_kinds) > 1:
        return []
    kind = target_kinds[0]
    if kind not in SLICE_KIND_CARDS:
        return []
    return card_identity_ids(_get_card_entities(kind, state, combat)[: _BY_KIND[kind].size])


# Action types already warned about dropped selections — truncation is never silent
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
        combat = state.combat  # hoisted once: FFI context accesses deep-clone
        pending_at = pending_virtual_int(state)  # likewise: one halt per state
        for action in legal_actions_batch[i]:
            # Fold the halt resolver onto its pool's virtual type (types.PendingType)
            if action.action_type is ActionType.EffectPendingResolve:
                if pending_at is None:
                    raise RuntimeError(
                        "EffectPendingResolve is legal but state.effect_pending is None"
                    )
                at = pending_at
            else:
                at = int(action.action_type)
            idxs = action.idxs
            target_kinds = ACTION_TYPE_POOL.get(at)
            if target_kinds is not None and idxs:
                # L1 slot: region-placed for the reward-card actions, the raw
                # engine index everywhere else
                if at in _AT_REWARD_CARD:
                    slot = reward_slot(action.action_type, idxs)
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
                # (RewardTakeCard's idxs[1] is a card-within-bundle, folded above)
                if at in _AT_MONSTER_TARGETED and len(idxs) == 2 and idxs[1] < MAX_MONSTERS:
                    if target_kinds[0] in SLICE_KIND_CARDS:
                        np_card_l2[i, slot, idxs[1]] = True
                    elif target_kinds[0] in SLICE_KIND_POTIONS:
                        np_potion_l2[i, slot, idxs[1]] = True
            np_action_type[i, at] = True  # ActionType: every legal kind the masks can express
        if not np_action_type[i].any():
            # Every legal action exceeded a pool cap — unrepresentable; fail loudly not NaN
            raise RuntimeError(
                f"All legal actions exceed encoder pool caps on focus "
                f"{derive_focus(state).name}: "
                f"{[(int(a.action_type), list(a.idxs)) for a in legal_actions_batch[i]]}"
            )

        # Dedup each type's l1 mask by identity pile (copies collapse); non-dedup pools return []
        for at, target_kinds in ACTION_TYPE_POOL.items():
            np_row = np_l1[at][i]
            if not np_row.any():
                continue
            ids = _pile_ids(state, combat, target_kinds)
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
