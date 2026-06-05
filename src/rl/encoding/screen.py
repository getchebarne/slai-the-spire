import numpy as np
import torch
from slai import GameState
from slai import Screen


_SCREEN_TO_IDX = {int(screen): i for i, screen in enumerate(Screen)}
_ENCODING_DIM_SCREEN = (
    len(Screen)            # Screen OHE
    + 1                    # Halted on a pending input
)


def _encode_screen_into(state: GameState, out: np.ndarray) -> None:
    # Screen OHE
    idx = _SCREEN_TO_IDX.get(int(state.screen))
    if idx is not None:
        out[idx] = 1.0

    # Halted on a pending input (combat sub-states collapse to Screen.Combat)
    out[len(Screen)] = float(state.pending is not None)


def encode_batch_screen(
    batch_state: list[GameState], device: torch.device
) -> torch.Tensor:
    batch_size = len(batch_state)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, _ENCODING_DIM_SCREEN), dtype=np.float32)

    for b, state in enumerate(batch_state):
        _encode_screen_into(state, x_out[b])

    return torch.from_numpy(x_out).to(device)
