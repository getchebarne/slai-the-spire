import numpy as np
import torch
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.constants import MAX_POTION_SLOTS


_POTION_NAME_TO_IDX = {potion_name: i for i, potion_name in enumerate(PotionName)}
_POTION_RARITY_TO_IDX = {potion_rarity: i for i, potion_rarity in enumerate(PotionRarity)}

ENCODING_DIM_POTION = (
    len(PotionName)         # Name OHE
    + len(PotionRarity)     # Rarity OHE
    + 1                     # Requires target
    + 1                     # Combat only
)


def encode_potion_into(potion: Potion, pos: int, out: np.ndarray) -> int:
    # Name OHE
    name_idx = _POTION_NAME_TO_IDX.get(int(potion.name))
    if name_idx is not None:
        out[pos + name_idx] = 1.0
    pos += len(PotionName)

    # Rarity OHE
    rarity_idx = _POTION_RARITY_TO_IDX.get(int(potion.rarity))
    if rarity_idx is not None:
        out[pos + rarity_idx] = 1.0
    pos += len(PotionRarity)

    # Scalars
    out[pos] = float(potion.requires_target)
    out[pos + 1] = float(potion.combat_only)
    pos += 2

    return pos


def encode_batch_potions(
    batch_potions: list[list[Potion]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
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
