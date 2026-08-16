import numpy as np
import torch
from slai import GameState


# Thermometer over the cumulative ascension rules: bit i = (ascension > i). Adjacent
# levels differ by exactly one bit — built-in transfer across curriculum steps — while
# each bit is a dedicated feature for one rule increment (the discrete-jump
# expressiveness a one-hot wants, without its orthogonal-neighbors cost). The scalar
# level is the bits' sum, so it is not encoded separately.
ENCODING_DIM_ASCENSION = 20


def encode_batch_ascension(
    batch_game_state: list[GameState], device: torch.device
) -> torch.Tensor:
    np_out = np.zeros((len(batch_game_state), ENCODING_DIM_ASCENSION), dtype=np.float32)
    for b, game_state in enumerate(batch_game_state):
        np_out[b, : game_state.ascension] = 1.0
    return torch.from_numpy(np_out).to(device)
