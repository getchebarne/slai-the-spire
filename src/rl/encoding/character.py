import numpy as np
import torch
from slai import Character

from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.encoding.modifier import encode_modifiers_into
from src.rl.encoding.modifier import get_encoding_dim_modifiers
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm


_INCOMING_DAMAGE_MAX = 150  # summed multi-monster turn; sqrt-scaled
_HEALTH_MAX = 80   # OHE bucket range (fixed for stable dim); the live max HP is a separate scalar
_HEALTH_MAX_CAP = 200  # sqrt cap for the max-HP magnitude scalar
_BLOCK_MAX = 30
_GOLD_MIN = 0
_GOLD_MAX = 499
_GOLD_LINEAR_SQRT_THRESHOLD = _GOLD_MIN # Pure sqrt
_GOLD_DIM = get_piecewise_dim(_GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD)
_ENCODING_DIM_CHARACTER = (
    get_encoding_dim_modifiers()                              # Modifiers OHE
    + get_encoding_dim_health_block(_HEALTH_MAX, _BLOCK_MAX)  # Health and block OHE and scalars
    + _GOLD_DIM                                               # Gold OHE
    + 1                                                       # Gold scalar
    + 1                                                       # Incoming damage
    + 1                                                       # Block >= incoming damage
    + 1                                                       # Incoming damage is lethal
    + 1                                                       # Health fraction of live max HP
    + 1                                                       # Max HP magnitude
    + 1                                                       # Net unblocked damage this turn
)

def _encode_character_into(
    character: Character, incoming_damage: int, out: np.ndarray
) -> None:
    # Initialize current position pointer
    pos = 0

    # Modifiers OHE
    pos = encode_modifiers_into(character.modifiers, pos, out)

    # Health and block OHE and scalars
    pos = encode_health_block_into(
        character.health, character.block, _HEALTH_MAX, _BLOCK_MAX, pos, out
    )

    # Gold OHE
    gold_bucket = get_piecewise_bucket(character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD)
    out[pos + gold_bucket] = 1.0
    pos += _GOLD_DIM

    # Scalars
    out[pos] = character.gold / _GOLD_MAX
    out[pos + 1] = get_sqrt_norm(incoming_damage, _INCOMING_DAMAGE_MAX)
    out[pos + 2] = float(character.block >= incoming_damage)
    out[pos + 3] = float(incoming_damage >= character.health + character.block)
    out[pos + 4] = character.health / max(character.health_max, 1)
    out[pos + 5] = get_sqrt_norm(character.health_max, _HEALTH_MAX_CAP)
    out[pos + 6] = get_sqrt_norm(max(incoming_damage - character.block, 0), _INCOMING_DAMAGE_MAX)


def encode_batch_character(
    batch_character: list[Character],
    batch_incoming_damage: list[int],
    device: torch.device,
) -> torch.Tensor:
    batch_size = len(batch_character)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, _ENCODING_DIM_CHARACTER), dtype=np.float32)

    for b, (character, incoming_damage) in enumerate(
        zip(batch_character, batch_incoming_damage)
    ):
        _encode_character_into(character, incoming_damage, x_out[b])

    return torch.from_numpy(x_out).to(device)
