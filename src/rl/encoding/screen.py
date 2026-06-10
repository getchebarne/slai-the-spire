import numpy as np
import torch
from slai import GameState
from slai import Screen
from slai import SelectionKind
from slai import members

from src.rl.constants import MAX_SIZE_HAND


_SCREEN_TO_IDX = {screen: i for i, screen in enumerate(members(Screen))}
# Input-requiring pending effects (kept local to avoid an encoding->action_space
# dependency). Combat/Event sub-states collapse to one Screen, so the pending kind is
# what lets the value head tell e.g. discard from retain.
_PENDING_EFFECTS = (
    "CardDiscard",
    "CardRetain",
    "CardSetupPick",
    "CardNightmarePick",
    "CardDiscoverPick",
    "CardPurge",
    "CardUpgrade",
    "CardDuplicate",
    "CardTransform",
)
_PENDING_TO_IDX = {name: i for i, name in enumerate(_PENDING_EFFECTS)}
# Normalized remaining pick count for an Input-selection halt; lets the value head
# tell "1 discard left" from "N left" over the same hand (discard-N re-halts per pick)
_COUNT_IDX = len(_SCREEN_TO_IDX) + len(_PENDING_TO_IDX)
_ENCODING_DIM_SCREEN = _COUNT_IDX + 1


def _encode_screen_into(state: GameState, out: np.ndarray) -> None:
    # Screen OHE
    out[_SCREEN_TO_IDX[state.screen]] = 1.0

    # Pending input kind OHE (all-zero = no pending) + normalized remaining pick count
    if state.pending is not None:
        out[len(_SCREEN_TO_IDX) + _PENDING_TO_IDX[type(state.pending).__name__]] = 1.0
        target = state.pending.target
        sk = target.selection_kind if target is not None else None
        if isinstance(sk, SelectionKind.Input):
            out[_COUNT_IDX] = sk.count / MAX_SIZE_HAND


def encode_batch_screen(
    batch_state: list[GameState], device: torch.device
) -> torch.Tensor:
    batch_size = len(batch_state)

    # Pre-allocate NumPy array
    x_out = np.zeros((batch_size, _ENCODING_DIM_SCREEN), dtype=np.float32)

    for b, state in enumerate(batch_state):
        _encode_screen_into(state, x_out[b])

    return torch.from_numpy(x_out).to(device)
