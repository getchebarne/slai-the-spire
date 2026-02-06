import math

import numpy as np
import torch

from src.game.const import MAX_MONSTERS
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4  # TODO: queries
from src.game.factory.monster.the_guardian import _HEALTH_MAX_ASC_9  # TODO: queries
from src.game.factory.monster.the_guardian import _WHIRLWIND_INSTANCES  # TODO: queries
from src.game.view.monster import ViewMonster
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.encoding.actor import get_encoding_dim_actor_modifiers


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_DAMAGE_MAX = _FIERCE_BASH_DAMAGE_ASC_4
_HEALTH_MAX = _HEALTH_MAX_ASC_9
_HEALTH_MIN = 1
_INSTANCES_MAX = _WHIRLWIND_INSTANCES
_MONSTER_NAMES = list(FACTORY_LIB_MONSTER.keys())
_NUM_MONSTER_NAMES = len(_MONSTER_NAMES)

# Threshold for piecewise linear-sqrt encoding
# Values 0 to threshold get exact (linear) buckets for fine-grained lethal detection
# Values above threshold get sqrt-compressed buckets
_LINEAR_SQRT_THRESHOLD = 18


def _get_piecewise_dim(min_val: int, max_val: int, threshold: int) -> int:
    """Calculate dimension for piecewise linear-sqrt encoding."""
    if max_val <= threshold:
        return max_val - min_val + 1  # Pure linear
    # Linear part: min_val to threshold
    linear_dim = threshold - min_val + 1
    # Sqrt part: floor(sqrt(threshold+1)) to floor(sqrt(max_val))
    sqrt_dim = int(math.sqrt(max_val)) - int(math.sqrt(threshold))
    return linear_dim + sqrt_dim


def _get_piecewise_bucket(value: int, min_val: int, threshold: int) -> int:
    """Get bucket index for piecewise linear-sqrt encoding."""
    value = max(value, min_val)  # Clamp to min
    if value <= threshold:
        return value - min_val
    # Above threshold: use sqrt buckets
    sqrt_threshold = int(math.sqrt(threshold))
    sqrt_value = int(math.sqrt(value))
    return (threshold - min_val + 1) + (sqrt_value - sqrt_threshold - 1)


# Pre-computed dimensions for piecewise encoding
_HEALTH_DIM = _get_piecewise_dim(_HEALTH_MIN, _HEALTH_MAX, _LINEAR_SQRT_THRESHOLD)
_BLOCK_DIM = _get_piecewise_dim(_BLOCK_MIN, _BLOCK_MAX, _LINEAR_SQRT_THRESHOLD)
_HP_BLOCK_DIM = _get_piecewise_dim(
    _HEALTH_MIN + _BLOCK_MIN, _HEALTH_MAX + _BLOCK_MAX, _LINEAR_SQRT_THRESHOLD
)
_DAMAGE_DIM = _get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)


def get_encoding_dim_monster() -> int:
    """Calculate monster encoding dimension (excludes modifiers, encoded separately)."""
    return (
        _NUM_MONSTER_NAMES  # Monster name one-hot
        + _HEALTH_DIM  # Health piecewise one-hot
        + _BLOCK_DIM  # Block piecewise one-hot
        + _HP_BLOCK_DIM  # HP+Block piecewise one-hot
        + _DAMAGE_DIM  # Intent damage piecewise one-hot
        + 5  # Intent scalars: damage, instances, block, buff, debuff
        + 3  # Scalars: health, block, hp+block
    )


_ENCODING_DIM_MONSTER = get_encoding_dim_monster()


def _encode_view_monster_into(out: np.ndarray, view_monster: ViewMonster) -> None:
    """Encode a monster directly into a pre-allocated numpy array."""
    pos = 0

    # Monster name one-hot
    idx_name = _MONSTER_NAMES.index(view_monster.name)
    out[idx_name] = 1.0
    pos += _NUM_MONSTER_NAMES

    # Health piecewise one-hot (linear for 0-threshold, sqrt above)
    health_bucket = _get_piecewise_bucket(
        view_monster.health_current, _HEALTH_MIN, _LINEAR_SQRT_THRESHOLD
    )
    out[pos + health_bucket] = 1.0
    pos += _HEALTH_DIM

    # Block piecewise one-hot
    block_bucket = _get_piecewise_bucket(
        view_monster.block_current, _BLOCK_MIN, _LINEAR_SQRT_THRESHOLD
    )
    out[pos + block_bucket] = 1.0
    pos += _BLOCK_DIM

    # HP+Block piecewise one-hot
    hp_block = view_monster.health_current + view_monster.block_current
    hp_block_bucket = _get_piecewise_bucket(
        hp_block, _HEALTH_MIN + _BLOCK_MIN, _LINEAR_SQRT_THRESHOLD
    )
    out[pos + hp_block_bucket] = 1.0
    pos += _HP_BLOCK_DIM

    # Intent
    damage = view_monster.intent.damage or 0

    # Damage piecewise one-hot
    damage_bucket = _get_piecewise_bucket(damage, 0, _LINEAR_SQRT_THRESHOLD)
    out[pos + damage_bucket] = 1.0
    pos += _DAMAGE_DIM

    # Intent scalars
    out[pos] = damage / _DAMAGE_MAX
    out[pos + 1] = (view_monster.intent.instances or 0) / _INSTANCES_MAX
    out[pos + 2] = float(view_monster.intent.block)
    out[pos + 3] = float(view_monster.intent.buff)
    out[pos + 4] = float(view_monster.intent.debuff_powerful)
    pos += 5

    # Scalars
    out[pos] = view_monster.health_current / _HEALTH_MAX
    out[pos + 1] = view_monster.block_current / _BLOCK_MAX
    out[pos + 2] = (view_monster.health_current + view_monster.block_current) / (
        _HEALTH_MAX + _BLOCK_MAX
    )


def encode_batch_view_monsters(
    batch_view_monster: list[list[ViewMonster]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Encode a batch of monster lists using NumPy pre-allocation.

    Returns:
        x_out: (B, MAX_MONSTERS, dim_monster) entity features (no modifiers)
        x_mask_pad: (B, MAX_MONSTERS) padding mask
        x_modifiers: (B, MAX_MONSTERS, dim_modifiers) modifier vectors
        outgoing_damages: list of total incoming damage per sample
    """
    batch_size = len(batch_view_monster)
    modifier_dim = get_encoding_dim_actor_modifiers()

    # Pre-allocate numpy arrays
    x_out = np.zeros((batch_size, MAX_MONSTERS, _ENCODING_DIM_MONSTER), dtype=np.float32)
    x_mask_pad = np.zeros((batch_size, MAX_MONSTERS), dtype=np.float32)
    x_modifiers = np.zeros((batch_size, MAX_MONSTERS, modifier_dim), dtype=np.float32)
    outgoing_damages = []

    for b, view_monsters in enumerate(batch_view_monster):
        outgoing_damage = 0.0
        for i, view_monster in enumerate(view_monsters):
            _encode_view_monster_into(x_out[b, i], view_monster)
            x_modifiers[b, i] = encode_view_actor_modifiers(view_monster.modifiers)
            x_mask_pad[b, i] = 1.0
            outgoing_damage += (view_monster.intent.damage or 0.0) * (
                view_monster.intent.instances or 1.0
            )
        outgoing_damages.append(outgoing_damage)

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_mask_pad).to(device),
        torch.from_numpy(x_modifiers).to(device),
        outgoing_damages,
    )
