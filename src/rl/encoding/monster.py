import numpy as np
import torch

from src.game.const import MAX_MONSTERS
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4  # TODO: queries
from src.game.factory.monster.the_guardian import _WHIRLWIND_INSTANCES  # TODO: queries
from src.game.view.monster import ViewMonster
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.encoding.actor import get_encoding_dim_actor_modifiers
from src.rl.encoding.health_block import _get_piecewise_bucket
from src.rl.encoding.health_block import _get_piecewise_dim
from src.rl.encoding.health_block import _LINEAR_SQRT_THRESHOLD
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block


_DAMAGE_MAX = _FIERCE_BASH_DAMAGE_ASC_4
_INSTANCES_MAX = _WHIRLWIND_INSTANCES
_MONSTER_NAMES = list(FACTORY_LIB_MONSTER.keys())
_NUM_MONSTER_NAMES = len(_MONSTER_NAMES)

_DAMAGE_DIM = _get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)


def get_encoding_dim_monster() -> int:
    """Calculate monster encoding dimension (excludes modifiers and health/block, encoded separately)."""
    return (
        _NUM_MONSTER_NAMES  # Monster name one-hot
        + _DAMAGE_DIM  # Intent damage piecewise one-hot
        + 5  # Intent scalars: damage, instances, block, buff, debuff
    )


_ENCODING_DIM_MONSTER = get_encoding_dim_monster()


def _encode_view_monster_into(out: np.ndarray, view_monster: ViewMonster) -> None:
    """Encode monster identity + intent into a pre-allocated numpy array."""
    pos = 0

    # Monster name one-hot
    idx_name = _MONSTER_NAMES.index(view_monster.name)
    out[idx_name] = 1.0
    pos += _NUM_MONSTER_NAMES

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


def encode_batch_view_monsters(
    batch_view_monster: list[list[ViewMonster]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[float]]:
    """Encode a batch of monster lists using NumPy pre-allocation.

    Returns:
        x_out: (B, MAX_MONSTERS, dim_monster) identity + intent features
        x_mask_pad: (B, MAX_MONSTERS) padding mask
        x_health_block: (B, MAX_MONSTERS, dim_health_block) shared health/block encoding
        x_modifiers: (B, MAX_MONSTERS, dim_modifiers) modifier vectors
        outgoing_damages: list of total incoming damage per sample
    """
    batch_size = len(batch_view_monster)
    modifier_dim = get_encoding_dim_actor_modifiers()
    health_block_dim = get_encoding_dim_health_block()

    # Pre-allocate numpy arrays
    x_out = np.zeros((batch_size, MAX_MONSTERS, _ENCODING_DIM_MONSTER), dtype=np.float32)
    x_mask_pad = np.zeros((batch_size, MAX_MONSTERS), dtype=np.float32)
    x_health_block = np.zeros((batch_size, MAX_MONSTERS, health_block_dim), dtype=np.float32)
    x_modifiers = np.zeros((batch_size, MAX_MONSTERS, modifier_dim), dtype=np.float32)
    outgoing_damages = []

    for b, view_monsters in enumerate(batch_view_monster):
        outgoing_damage = 0.0
        for i, view_monster in enumerate(view_monsters):
            _encode_view_monster_into(x_out[b, i], view_monster)
            encode_health_block_into(
                x_health_block[b, i],
                view_monster.health_current,
                view_monster.block_current,
            )
            x_modifiers[b, i] = encode_view_actor_modifiers(view_monster.modifiers)
            x_mask_pad[b, i] = 1.0
            outgoing_damage += (view_monster.intent.damage or 0.0) * (
                view_monster.intent.instances or 1.0
            )
        outgoing_damages.append(outgoing_damage)

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_mask_pad).to(device),
        torch.from_numpy(x_health_block).to(device),
        torch.from_numpy(x_modifiers).to(device),
        outgoing_damages,
    )
