import math

import numpy as np
import torch

from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4
from src.game.view.character import ViewCharacter
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.encoding.actor import get_encoding_dim_actor_modifiers


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_HEALTH_MAX = 70  # TODO: will be dynamic in the future
_HEALTH_MIN = 1

# Pre-computed sqrt bounds for one-hot encoding (AlphaStar-style compression)
_SQRT_HEALTH_MIN = int(math.sqrt(_HEALTH_MIN))
_SQRT_HEALTH_MAX = int(math.sqrt(_HEALTH_MAX))
_SQRT_BLOCK_MIN = int(math.sqrt(_BLOCK_MIN))
_SQRT_BLOCK_MAX = int(math.sqrt(_BLOCK_MAX))
_SQRT_HP_BLOCK_MIN = int(math.sqrt(_HEALTH_MIN + _BLOCK_MIN))
_SQRT_HP_BLOCK_MAX = int(math.sqrt(_HEALTH_MAX + _BLOCK_MAX))


def get_encoding_dim_character() -> int:
    """Calculate character encoding dimension."""
    return (
        (_SQRT_HEALTH_MAX - _SQRT_HEALTH_MIN + 1)  # Health sqrt one-hot
        + (_SQRT_BLOCK_MAX - _SQRT_BLOCK_MIN + 1)  # Block sqrt one-hot
        + (_SQRT_HP_BLOCK_MAX - _SQRT_HP_BLOCK_MIN + 1)  # HP+Block sqrt one-hot
        + get_encoding_dim_actor_modifiers()  # Modifiers
        + 5  # Scalars: health, block, hp+block, incoming_damage, blocked
    )


_ENCODING_DIM_CHARACTER = get_encoding_dim_character()


def _encode_view_character_into(
    out: np.ndarray, view_character: ViewCharacter, incoming_damage: int
) -> None:
    """Encode a character directly into a pre-allocated numpy array."""
    pos = 0

    # Health sqrt one-hot
    sqrt_health = int(math.sqrt(view_character.health_current))
    sqrt_health = max(min(sqrt_health, _SQRT_HEALTH_MAX), _SQRT_HEALTH_MIN)
    out[pos + sqrt_health - _SQRT_HEALTH_MIN] = 1.0
    pos += _SQRT_HEALTH_MAX - _SQRT_HEALTH_MIN + 1

    # Block sqrt one-hot
    sqrt_block = int(math.sqrt(view_character.block_current))
    sqrt_block = max(min(sqrt_block, _SQRT_BLOCK_MAX), _SQRT_BLOCK_MIN)
    out[pos + sqrt_block - _SQRT_BLOCK_MIN] = 1.0
    pos += _SQRT_BLOCK_MAX - _SQRT_BLOCK_MIN + 1

    # HP+Block sqrt one-hot
    sqrt_hp_block = int(math.sqrt(view_character.health_current + view_character.block_current))
    sqrt_hp_block = max(min(sqrt_hp_block, _SQRT_HP_BLOCK_MAX), _SQRT_HP_BLOCK_MIN)
    out[pos + sqrt_hp_block - _SQRT_HP_BLOCK_MIN] = 1.0
    pos += _SQRT_HP_BLOCK_MAX - _SQRT_HP_BLOCK_MIN + 1

    # Modifiers
    modifiers_list = encode_view_actor_modifiers(view_character.modifiers)
    modifier_dim = len(modifiers_list)
    out[pos : pos + modifier_dim] = modifiers_list
    pos += modifier_dim

    # Scalars
    out[pos] = view_character.health_current / _HEALTH_MAX
    out[pos + 1] = view_character.block_current / _BLOCK_MAX
    out[pos + 2] = (view_character.health_current + view_character.block_current) / (_HEALTH_MAX + _BLOCK_MAX)
    out[pos + 3] = incoming_damage / _FIERCE_BASH_DAMAGE_ASC_4
    out[pos + 4] = float(view_character.block_current >= incoming_damage)


def encode_batch_view_character(
    batch_view_character: list[ViewCharacter],
    batch_incoming_damage: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Encode a batch of characters using NumPy pre-allocation.
    
    Returns: (B, dim_character) tensor
    """
    batch_size = len(batch_view_character)

    # Pre-allocate numpy array (2D: batch x features)
    x_out = np.zeros((batch_size, _ENCODING_DIM_CHARACTER), dtype=np.float32)

    for b, (view_character, incoming_damage) in enumerate(
        zip(batch_view_character, batch_incoming_damage)
    ):
        _encode_view_character_into(x_out[b], view_character, incoming_damage)

    return torch.from_numpy(x_out).to(device)
