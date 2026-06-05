import numpy as np
import torch
from slai import Relic
from slai import RelicName

from src.rl.constants import MAX_RELICS


_RELIC_NAME_TO_IDX = {relic_name: i for i, relic_name in enumerate(RelicName)}
_COUNTER_MAX = 9

ENCODING_DIM_RELIC = (
    len(RelicName)         # Name OHE
    + 1                    # Used up
    + 1                    # Counter scalar
)


def encode_relic_into(relic: Relic, pos: int, out: np.ndarray) -> int:
    # Name OHE
    idx = _RELIC_NAME_TO_IDX.get(int(relic.name))
    if idx is not None:
        out[pos + idx] = 1.0

    # Scalars
    out[pos + len(RelicName)] = float(relic.used_up)
    out[pos + len(RelicName) + 1] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    return pos + ENCODING_DIM_RELIC


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
