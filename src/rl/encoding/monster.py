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

# Pre-computed sqrt bounds for one-hot encoding
_SQRT_HEALTH_MIN = int(math.sqrt(_HEALTH_MIN))
_SQRT_HEALTH_MAX = int(math.sqrt(_HEALTH_MAX))
_SQRT_BLOCK_MIN = int(math.sqrt(_BLOCK_MIN))
_SQRT_BLOCK_MAX = int(math.sqrt(_BLOCK_MAX))
_SQRT_HP_BLOCK_MIN = int(math.sqrt(_HEALTH_MIN + _BLOCK_MIN))
_SQRT_HP_BLOCK_MAX = int(math.sqrt(_HEALTH_MAX + _BLOCK_MAX))

# Sqrt bounds for intent damage
_SQRT_DAMAGE_MIN = 0
_SQRT_DAMAGE_MAX = int(math.sqrt(_DAMAGE_MAX))
_SQRT_DAMAGE_DIM = _SQRT_DAMAGE_MAX - _SQRT_DAMAGE_MIN + 1


def get_encoding_dim_monster() -> int:
    """Calculate monster encoding dimension."""
    return (
        _NUM_MONSTER_NAMES  # Monster name one-hot
        + (_SQRT_HEALTH_MAX - _SQRT_HEALTH_MIN + 1)  # Health sqrt one-hot
        + (_SQRT_BLOCK_MAX - _SQRT_BLOCK_MIN + 1)  # Block sqrt one-hot
        + (_SQRT_HP_BLOCK_MAX - _SQRT_HP_BLOCK_MIN + 1)  # HP+Block sqrt one-hot
        + get_encoding_dim_actor_modifiers()  # Modifiers
        + _SQRT_DAMAGE_DIM  # Intent damage sqrt one-hot
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

    # Health sqrt one-hot
    sqrt_health = int(math.sqrt(view_monster.health_current))
    sqrt_health = max(min(sqrt_health, _SQRT_HEALTH_MAX), _SQRT_HEALTH_MIN)
    out[pos + sqrt_health - _SQRT_HEALTH_MIN] = 1.0
    pos += _SQRT_HEALTH_MAX - _SQRT_HEALTH_MIN + 1

    # Block sqrt one-hot
    sqrt_block = int(math.sqrt(view_monster.block_current))
    sqrt_block = max(min(sqrt_block, _SQRT_BLOCK_MAX), _SQRT_BLOCK_MIN)
    out[pos + sqrt_block - _SQRT_BLOCK_MIN] = 1.0
    pos += _SQRT_BLOCK_MAX - _SQRT_BLOCK_MIN + 1

    # HP+Block sqrt one-hot
    sqrt_hp_block = int(math.sqrt(view_monster.health_current + view_monster.block_current))
    sqrt_hp_block = max(min(sqrt_hp_block, _SQRT_HP_BLOCK_MAX), _SQRT_HP_BLOCK_MIN)
    out[pos + sqrt_hp_block - _SQRT_HP_BLOCK_MIN] = 1.0
    pos += _SQRT_HP_BLOCK_MAX - _SQRT_HP_BLOCK_MIN + 1

    # Modifiers (using list-based helper, then copy)
    modifiers_list = encode_view_actor_modifiers(view_monster.modifiers)
    modifier_dim = len(modifiers_list)
    out[pos : pos + modifier_dim] = modifiers_list
    pos += modifier_dim

    # Intent
    damage = view_monster.intent.damage or 0

    # Damage sqrt one-hot
    sqrt_damage = int(math.sqrt(damage))
    sqrt_damage = max(min(sqrt_damage, _SQRT_DAMAGE_MAX), _SQRT_DAMAGE_MIN)
    out[pos + sqrt_damage - _SQRT_DAMAGE_MIN] = 1.0
    pos += _SQRT_DAMAGE_DIM

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
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Encode a batch of monster lists using NumPy pre-allocation."""
    batch_size = len(batch_view_monster)

    # Pre-allocate numpy arrays
    x_out = np.zeros((batch_size, MAX_MONSTERS, _ENCODING_DIM_MONSTER), dtype=np.float32)
    x_mask_pad = np.zeros((batch_size, MAX_MONSTERS), dtype=np.float32)
    outgoing_damages = []

    for b, view_monsters in enumerate(batch_view_monster):
        outgoing_damage = 0.0
        for i, view_monster in enumerate(view_monsters):
            _encode_view_monster_into(x_out[b, i], view_monster)
            x_mask_pad[b, i] = 1.0
            outgoing_damage += (view_monster.intent.damage or 0.0) * (
                view_monster.intent.instances or 1.0
            )
        outgoing_damages.append(outgoing_damage)

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_mask_pad).to(device),
        outgoing_damages,
    )
