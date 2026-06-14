import numpy as np
import torch
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import members

from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.index import CLASS_SEGMENTS
from src.rl.index import CLASS_SLICE
from src.rl.index import NUM_CLASS_TOKENS
from src.rl.index import EntityClass


_RELIC_NAME_TO_IDX = {relic_name: i for i, relic_name in enumerate(members(RelicName))}
_COUNTER_MAX = 9

ENCODING_DIM_RELIC = (
    len(_RELIC_NAME_TO_IDX)  # Name OHE
    + 1  # Used up
    + 1  # Counter scalar
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
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every relic segment (registry RELIC class) into one concatenated
    (B, N_RELICS, ENCODING_DIM_RELIC) tensor + mask; segments live at their
    index.CLASS_SLICE positions."""
    batch_size = len(batch_game_state)
    num_tokens = NUM_CLASS_TOKENS[EntityClass.RELIC]

    # Pre-allocate NumPy arrays
    x_out = np.zeros((batch_size, num_tokens, ENCODING_DIM_RELIC), dtype=np.float32)
    x_pad = np.zeros((batch_size, num_tokens), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        for spec in CLASS_SEGMENTS[EntityClass.RELIC]:
            offset = CLASS_SLICE[spec.segment].start
            for i, relic in enumerate(spec.getter(game_state)):
                encode_relic_into(relic, 0, x_out[b, offset + i])
                x_pad[b, offset + i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
