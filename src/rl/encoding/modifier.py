import math
import warnings

import numpy as np
import torch
from slai import GameState
from slai import Modifier
from slai import ModifierKind

from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import MAX_MODIFIER_ROWS_STATE
from src.rl.constants import MAX_MONSTERS
from src.rl.types import EntityKind
from src.rl.types import X_Modifiers
from src.rl.utils import get_sqrt_norm


_MODIFIER_KIND_TO_IDX = {
    modifier_kind: int(modifier_kind) for modifier_kind in ModifierKind.members()
}
_STACKS_MAX = 99
# Per-kind caps over the default: Poison past the widest monster HP kills on its next tick
_STACKS_MAX_OF_KIND = {ModifierKind.Poison: HEALTH_MAX_ENTITY}
_OFF_MODIFIER_BUFF = len(_MODIFIER_KIND_TO_IDX)
_OFF_MODIFIER_STACKS = _OFF_MODIFIER_BUFF + 1
_OFF_MODIFIER_STACKS_NEGATIVE = _OFF_MODIFIER_STACKS + 1
_OFF_MODIFIER_STACKS_OHE = _OFF_MODIFIER_STACKS_NEGATIVE + 1
_STACKS_BUCKET_DIM = int(math.sqrt(max(_STACKS_MAX, *_STACKS_MAX_OF_KIND.values()))) + 1

ENCODING_DIM_MODIFIER = _OFF_MODIFIER_STACKS_OHE + _STACKS_BUCKET_DIM


def encode_modifier_row_into(modifier: Modifier, out: np.ndarray) -> None:
    # Modifier kind
    out[_MODIFIER_KIND_TO_IDX[modifier.kind]] = 1.0

    # Buff or debuff
    out[_OFF_MODIFIER_BUFF] = float(modifier.kind.is_buff)

    # Stacks: signed float + sign bit + magnitude-only sqrt buckets
    stacks_max = _STACKS_MAX_OF_KIND.get(modifier.kind, _STACKS_MAX)
    out[_OFF_MODIFIER_STACKS] = get_sqrt_norm(modifier.stacks, stacks_max)
    out[_OFF_MODIFIER_STACKS_NEGATIVE] = float(modifier.stacks < 0)
    out[_OFF_MODIFIER_STACKS_OHE + int(math.sqrt(min(abs(modifier.stacks), stacks_max)))] = 1.0


# Row families that have already warned about overflow — truncation is never silent
_WARNED_OVERFLOW: set[str] = set()


def encode_batch_modifiers(game_states: list[GameState], device: torch.device) -> X_Modifiers:
    """Every state's Modifier rows in one (B, N, D) block with their owner keys: the
    Character's rows (offset 0), then each Monster's (offset = its slot)."""
    batch_size = len(game_states)
    np_out = np.zeros(
        (batch_size, MAX_MODIFIER_ROWS_STATE, ENCODING_DIM_MODIFIER), dtype=np.float32
    )
    np_entity_kind = np.full((batch_size, MAX_MODIFIER_ROWS_STATE), -1, dtype=np.int64)
    np_offset = np.zeros((batch_size, MAX_MODIFIER_ROWS_STATE), dtype=np.int64)

    idx_used = 0
    for b, game_state in enumerate(game_states):
        combat = game_state.combat
        monsters = combat.monsters[:MAX_MONSTERS] if combat is not None else []
        owners = [(EntityKind.CHARACTER, 0, game_state.character.modifiers)]
        owners += [
            (EntityKind.MONSTER, i, monster.modifiers) for i, monster in enumerate(monsters)
        ]

        idx = 0
        for entity_kind, w, modifiers in owners:
            for modifier in modifiers:
                if idx >= MAX_MODIFIER_ROWS_STATE:
                    if "modifiers" not in _WARNED_OVERFLOW:
                        _WARNED_OVERFLOW.add("modifiers")
                        warnings.warn(
                            f"modifier rows exceeded {MAX_MODIFIER_ROWS_STATE} in one state; "
                            "dropped (constants.MAX_MODIFIER_ROWS_STATE)"
                        )
                    break

                encode_modifier_row_into(modifier, np_out[b, idx])
                np_entity_kind[b, idx] = entity_kind
                np_offset[b, idx] = w
                idx += 1
        idx_used = max(idx_used, idx)

    # The batch's used rows only: the transport pads back to MAX_MODIFIER_ROWS_STATE
    return X_Modifiers(
        x=torch.from_numpy(np.ascontiguousarray(np_out[:, :idx_used])).to(device),
        x_entity_kind=torch.from_numpy(np.ascontiguousarray(np_entity_kind[:, :idx_used])).to(
            device
        ),
        x_offset=torch.from_numpy(np.ascontiguousarray(np_offset[:, :idx_used])).to(device),
        batch_size=[batch_size],
    )
