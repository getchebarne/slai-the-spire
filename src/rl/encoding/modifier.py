import math
import warnings
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState
from slai import Modifier
from slai import ModifierKind

from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import MAX_MODIFIER_ROWS_STATE
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.encoding.sparse import owned_sparse
from src.rl.types import SliceKind
from src.rl.types import X_OwnedSparse
from src.rl.types import make_channel_allocator


_MODIFIER_KIND_TO_IDX = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
_STACKS_MAX = 99

# Poison past the widest Monster HP kills on its next tick
_STACKS_MAX_OF_MODIFIER_KIND = {ModifierKind.Poison: HEALTH_MAX_ENTITY}
_STACKS_BUCKET_DIM = int(math.sqrt(max(_STACKS_MAX, *_STACKS_MAX_OF_MODIFIER_KIND.values()))) + 1


class ModifierChannel(StrEnum):
    KIND = auto()
    BUFF = auto()
    STACKS = auto()
    STACKS_NEGATIVE = auto()
    STACKS_OHE = auto()


CHANNEL_LAYOUT_MODIFIER, _alloc = make_channel_allocator()
_OFF_MODIFIER_KIND = _alloc(ModifierChannel.KIND, len(_MODIFIER_KIND_TO_IDX))
_OFF_MODIFIER_BUFF = _alloc(ModifierChannel.BUFF)
_OFF_MODIFIER_STACKS = _alloc(ModifierChannel.STACKS)
_OFF_MODIFIER_STACKS_NEGATIVE = _alloc(ModifierChannel.STACKS_NEGATIVE)
_OFF_MODIFIER_STACKS_OHE = _alloc(ModifierChannel.STACKS_OHE, _STACKS_BUCKET_DIM)

ENCODING_DIM_MODIFIER = sum(w for _, w in CHANNEL_LAYOUT_MODIFIER)

# Most nonzero columns a modifier row holds: one per channel, each a one-hot or a scalar
ENCODING_SPARSE_MODIFIER = len(CHANNEL_LAYOUT_MODIFIER)


def encode_modifier_row_into(modifier: Modifier, out: np.ndarray) -> None:
    # Modifier kind
    out[_OFF_MODIFIER_KIND + _MODIFIER_KIND_TO_IDX[modifier.kind]] = 1.0

    # Buff or debuff
    out[_OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only sqrt buckets
    stacks_max = _STACKS_MAX_OF_MODIFIER_KIND.get(modifier.kind, _STACKS_MAX)
    out[_OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, stacks_max)
    out[_OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    out[_OFF_MODIFIER_STACKS_OHE + int(math.sqrt(min(abs(modifier.stacks), stacks_max)))] = 1.0


# Row families that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()


def encode_batch_modifiers(
    game_states: list[GameState], device: torch.device
) -> tuple[X_OwnedSparse, torch.Tensor]:
    """Every state's Modifier rows in one (N, D) block, held sparse, every state's after the
    previous state's, with their owner keys: the Character's rows (CHARACTER, 0), then each
    Monster's (MONSTERS, its slot); and each state's (B,) row count."""
    batch_rows: list[list[tuple[SliceKind, int, Modifier]]] = []
    for game_state in game_states:
        combat = game_state.combat
        if combat is None:
            batch_rows.append([])
            continue

        monsters = combat.monsters
        owners = [(SliceKind.CHARACTER, 0, game_state.character.modifiers)]
        owners += [
            (SliceKind.MONSTERS, w, monster.modifiers) for w, monster in enumerate(monsters)
        ]
        batch_rows.append(
            [
                (slice_kind, w, modifier)
                for slice_kind, w, modifiers in owners
                for modifier in modifiers
            ]
        )

    # Each state's rows cut to MAX_MODIFIER_ROWS_STATE
    if "modifiers" not in _WARNED_OVERFLOW and any(
        len(rows) > MAX_MODIFIER_ROWS_STATE for rows in batch_rows
    ):
        _WARNED_OVERFLOW.add("modifiers")
        warnings.warn(
            f"modifier rows exceeded {MAX_MODIFIER_ROWS_STATE} in one state; "
            "dropped (constants.MAX_MODIFIER_ROWS_STATE)"
        )
    np_count = np.array(
        [min(len(rows), MAX_MODIFIER_ROWS_STATE) for rows in batch_rows], dtype=np.int64
    )
    np_out = np.zeros((np_count.sum(), ENCODING_DIM_MODIFIER), dtype=np.float32)
    np_owner_slice_kind = np.empty(np_count.sum(), dtype=np.int64)
    np_owner_rank = np.empty(np_count.sum(), dtype=np.int64)

    row = 0
    for rows, count in zip(batch_rows, np_count):
        for slice_kind, w, modifier in rows[:count]:
            encode_modifier_row_into(modifier, np_out[row])
            np_owner_slice_kind[row] = slice_kind
            np_owner_rank[row] = w
            row += 1

    return (
        owned_sparse(np_out, np_owner_slice_kind, np_owner_rank, ENCODING_SPARSE_MODIFIER, device),
        torch.from_numpy(np_count).to(device),
    )
