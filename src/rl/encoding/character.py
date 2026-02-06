import numpy as np
import torch

from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4
from src.game.view.character import ViewCharacter
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.encoding.actor import get_encoding_dim_actor_modifiers
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block


def get_encoding_dim_character() -> int:
    """Calculate character encoding dimension (excludes modifiers and health/block, encoded separately)."""
    return 2  # incoming_damage, blocked


_ENCODING_DIM_CHARACTER = get_encoding_dim_character()


def _encode_view_character_into(
    out: np.ndarray, view_character: ViewCharacter, incoming_damage: int
) -> None:
    """Encode character-specific features (survivability) into a pre-allocated numpy array."""
    out[0] = incoming_damage / _FIERCE_BASH_DAMAGE_ASC_4
    out[1] = float(view_character.block_current >= incoming_damage)


def encode_batch_view_character(
    batch_view_character: list[ViewCharacter],
    batch_incoming_damage: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode a batch of characters using NumPy pre-allocation.

    Returns:
        x_out: (B, dim_character) character-specific features (survivability)
        x_health_block: (B, dim_health_block) shared health/block encoding
        x_modifiers: (B, dim_modifiers) modifier vectors
    """
    batch_size = len(batch_view_character)
    modifier_dim = get_encoding_dim_actor_modifiers()
    health_block_dim = get_encoding_dim_health_block()

    # Pre-allocate numpy arrays
    x_out = np.zeros((batch_size, _ENCODING_DIM_CHARACTER), dtype=np.float32)
    x_health_block = np.zeros((batch_size, health_block_dim), dtype=np.float32)
    x_modifiers = np.zeros((batch_size, modifier_dim), dtype=np.float32)

    for b, (view_character, incoming_damage) in enumerate(
        zip(batch_view_character, batch_incoming_damage)
    ):
        _encode_view_character_into(x_out[b], view_character, incoming_damage)
        encode_health_block_into(
            x_health_block[b],
            view_character.health_current,
            view_character.block_current,
        )
        x_modifiers[b] = encode_view_actor_modifiers(view_character.modifiers)

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_health_block).to(device),
        torch.from_numpy(x_modifiers).to(device),
    )
