from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import GameState

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.sparse import owned_sparse
from src.rl.types import SliceKind
from src.rl.types import X_OwnedSparse
from src.rl.types import make_channel_allocator


_HEALTH_MIN = 1
_BLOCK_MIN = 0
_TOTAL_MIN = _HEALTH_MIN + _BLOCK_MIN
_TOTAL_MAX = HEALTH_MAX_ENTITY + BLOCK_MAX_ENTITY
_LINEAR_SQRT_THRESHOLD = 25
_HEALTH_OHE_DIM = get_piecewise_dim(_HEALTH_MIN, HEALTH_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_BLOCK_OHE_DIM = get_piecewise_dim(_BLOCK_MIN, BLOCK_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD)
_TOTAL_OHE_DIM = get_piecewise_dim(_TOTAL_MIN, _TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)


class HealthBlockChannel(StrEnum):
    HEALTH_OHE = auto()
    BLOCK_OHE = auto()
    TOTAL_OHE = auto()
    HEALTH_SCALAR = auto()
    BLOCK_SCALAR = auto()
    TOTAL_SCALAR = auto()


CHANNEL_LAYOUT_HEALTH_BLOCK, _alloc = make_channel_allocator()
OFF_HEALTH_BLOCK_HEALTH = _alloc(HealthBlockChannel.HEALTH_OHE, _HEALTH_OHE_DIM)
OFF_HEALTH_BLOCK_BLOCK = _alloc(HealthBlockChannel.BLOCK_OHE, _BLOCK_OHE_DIM)
OFF_HEALTH_BLOCK_TOTAL = _alloc(HealthBlockChannel.TOTAL_OHE, _TOTAL_OHE_DIM)
OFF_HEALTH_BLOCK_HEALTH_SCALAR = _alloc(HealthBlockChannel.HEALTH_SCALAR)
OFF_HEALTH_BLOCK_BLOCK_SCALAR = _alloc(HealthBlockChannel.BLOCK_SCALAR)
OFF_HEALTH_BLOCK_TOTAL_SCALAR = _alloc(HealthBlockChannel.TOTAL_SCALAR)

# The one width every combatant builds: EntityProjector projects it with ONE shared block
ENCODING_DIM_HEALTH_BLOCK = sum(w for _, w in CHANNEL_LAYOUT_HEALTH_BLOCK)

# Most nonzero columns a health/block row holds: one per channel, each a one-hot or a scalar
ENCODING_SPARSE_HEALTH_BLOCK = len(CHANNEL_LAYOUT_HEALTH_BLOCK)


def _encode_health_block_into(health: int, block: int, out: np.ndarray) -> None:
    total = health + block

    health_bucket = get_piecewise_bucket(
        health, _HEALTH_MIN, HEALTH_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_HEALTH + health_bucket] = 1.0
    block_bucket = get_piecewise_bucket(
        block, _BLOCK_MIN, BLOCK_MAX_ENTITY, _LINEAR_SQRT_THRESHOLD
    )
    out[OFF_HEALTH_BLOCK_BLOCK + block_bucket] = 1.0
    total_bucket = get_piecewise_bucket(total, _TOTAL_MIN, _TOTAL_MAX, _LINEAR_SQRT_THRESHOLD)
    out[OFF_HEALTH_BLOCK_TOTAL + total_bucket] = 1.0

    # Scalars
    out[OFF_HEALTH_BLOCK_HEALTH_SCALAR] = min(health / HEALTH_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_BLOCK_SCALAR] = min(block / BLOCK_MAX_ENTITY, 1.0)
    out[OFF_HEALTH_BLOCK_TOTAL_SCALAR] = min(total / _TOTAL_MAX, 1.0)


def encode_batch_health_block(
    game_states: list[GameState], device: torch.device
) -> tuple[X_OwnedSparse, torch.Tensor]:
    """Every state's health/block rows as one (N, dim) block, held sparse, every state's after the
    previous state's, with their owner keys: each Monster's row (MONSTERS, its slot), then the
    Character's (CHARACTER, 0); and each state's (B,) row count, its monsters + 1."""
    batch_monsters = []
    for game_state in game_states:
        combat = game_state.combat  # hoisted: each read clones across the FFI
        batch_monsters.append(combat.monsters if combat is not None else [])

    np_count = np.array([len(monsters) + 1 for monsters in batch_monsters], dtype=np.int64)
    np_out = np.zeros((np_count.sum(), ENCODING_DIM_HEALTH_BLOCK), dtype=np.float32)
    np_owner_slice_kind = np.empty(np_count.sum(), dtype=np.int64)
    np_owner_rank = np.zeros(np_count.sum(), dtype=np.int64)

    row = 0
    for game_state, monsters in zip(game_states, batch_monsters):
        for k, monster in enumerate(monsters):
            _encode_health_block_into(monster.health, monster.block, np_out[row])
            np_owner_slice_kind[row] = SliceKind.MONSTERS
            np_owner_rank[row] = k
            row += 1
        character = game_state.character
        _encode_health_block_into(character.health, character.block, np_out[row])
        np_owner_slice_kind[row] = SliceKind.CHARACTER
        row += 1

    return (
        owned_sparse(
            np_out, np_owner_slice_kind, np_owner_rank, ENCODING_SPARSE_HEALTH_BLOCK, device
        ),
        torch.from_numpy(np_count).to(device),
    )
