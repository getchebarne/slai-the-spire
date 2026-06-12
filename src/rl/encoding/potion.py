import numpy as np
import torch
from slai import Potion
from slai import PotionName
from slai import PotionRarity
from slai import members

from src.rl.constants import MAX_POTION_SLOTS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into


_POTION_NAME_TO_IDX = {potion_name: i for i, potion_name in enumerate(members(PotionName))}
_POTION_RARITY_TO_IDX = {potion_rarity: i for i, potion_rarity in enumerate(members(PotionRarity))}

ENCODING_DIM_POTION = (
    len(_POTION_NAME_TO_IDX)  # Name OHE
    + len(_POTION_RARITY_TO_IDX)  # Rarity OHE
    + 1  # Requires target
    + 1  # Combat only
    + ENCODING_DIM_EFFECTS  # Per-EffectKind effect blocks
)


def encode_potion_into(potion: Potion, pos: int, out: np.ndarray) -> int:
    # Name OHE
    out[pos + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    pos += len(_POTION_NAME_TO_IDX)

    # Rarity OHE
    out[pos + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    pos += len(_POTION_RARITY_TO_IDX)

    # Scalars
    out[pos] = float(potion.requires_target)
    out[pos + 1] = float(potion.combat_only)
    pos += 2

    # Per-EffectKind effect blocks
    return encode_effects_into(potion.effects, pos, out)


def encode_batch_potions(
    batch_potions: list[list[Potion]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (encodings, padding mask) over the belt. Targeting is no longer derived
    here — the L3 target mask comes from the engine's legal actions (masks.py)."""
    batch_size = len(batch_potions)

    # Pre-allocate NumPy arrays
    x_out = np.zeros((batch_size, MAX_POTION_SLOTS, ENCODING_DIM_POTION), dtype=np.float32)
    x_pad = np.zeros((batch_size, MAX_POTION_SLOTS), dtype=bool)

    for b, potions in enumerate(batch_potions):
        for i, potion in enumerate(potions):
            if potion is None:
                continue

            encode_potion_into(potion, 0, x_out[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
