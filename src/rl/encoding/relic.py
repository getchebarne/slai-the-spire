import numpy as np
import torch
from slai import Relic
from slai import RelicName
from slai import members

from src.rl.constants import MAX_RELICS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into


_RELIC_NAME_TO_IDX = {relic_name: i for i, relic_name in enumerate(members(RelicName))}
_COUNTER_MAX = 9

ENCODING_DIM_RELIC = (
    len(_RELIC_NAME_TO_IDX)         # Name OHE
    + 1                    # Used up
    + 1                    # Counter scalar
    + ENCODING_DIM_EFFECTS  # Combat-start effect blocks (trigger timing rides on the name)
)


def encode_relic_into(relic: Relic, pos: int, out: np.ndarray) -> int:
    # Name OHE
    out[pos + _RELIC_NAME_TO_IDX[relic.name]] = 1.0
    pos += len(_RELIC_NAME_TO_IDX)

    # Scalars
    out[pos] = float(relic.used_up)
    out[pos + 1] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX
    pos += 2

    # Per-EffectKind effect blocks
    return encode_effects_into(relic.effects_on_combat_start, pos, out)


def encode_batch_relics(
    batch_relics: list[list[Relic]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(batch_relics)

    # Pre-allocate NumPy arrays
    x_out = np.zeros((batch_size, MAX_RELICS, ENCODING_DIM_RELIC), dtype=np.float32)
    x_pad = np.zeros((batch_size, MAX_RELICS), dtype=bool)

    for b, relics in enumerate(batch_relics):
        for i, relic in enumerate(relics):
            encode_relic_into(relic, 0, x_out[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
