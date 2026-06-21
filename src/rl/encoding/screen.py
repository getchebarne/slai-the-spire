import numpy as np
import torch
from slai import CandidatePool
from slai import Effect
from slai import GameState
from slai import Screen
from slai import SelectionKind
from slai import members

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_HAND


_SCREEN_TO_IDX = {screen: i for i, screen in enumerate(members(Screen))}
_EFFECT_PENDING = (
    Effect.CardDiscard,
    Effect.CardRetain,
    Effect.CardSetupPick,
    Effect.CardNightmarePick,
    Effect.CardDiscoverPick,
    Effect.CardPurge,
    Effect.CardUpgrade,
    Effect.CardDuplicate,
    Effect.CardTransform,
)
_EFFECT_PENDING_TO_IDX = {cls: i for i, cls in enumerate(_EFFECT_PENDING)}
_COUNT_IDX = len(_SCREEN_TO_IDX) + len(_EFFECT_PENDING_TO_IDX)
ENCODING_DIM_SCREEN = _COUNT_IDX + 1

# Pool the remaining-pick count is normalized against
_INPUT_POOL_CAP = {
    CandidatePool.Hand: MAX_SIZE_HAND,
    CandidatePool.Deck: MAX_SIZE_DECK,
    CandidatePool.Discover: MAX_SIZE_DISCOVER,
}


def _encode_screen_into(state: GameState, out: np.ndarray) -> None:
    # Screen OHE
    out[_SCREEN_TO_IDX[state.screen]] = 1.0

    # Pending input kind OHE (all-zero = no pending) + normalized remaining pick count
    if state.pending is not None:
        out[len(_SCREEN_TO_IDX) + _EFFECT_PENDING_TO_IDX[type(state.pending)]] = 1.0
        target = state.pending.target
        selection_kind = target.selection_kind if target is not None else None
        if isinstance(selection_kind, SelectionKind.Input):
            cap = _INPUT_POOL_CAP.get(type(target.candidate_pool))
            if cap is None:
                raise ValueError(
                    f"Input selection from unexpected pool {type(target.candidate_pool).__name__}"
                    f" (pending {type(state.pending).__name__})"
                )

            out[_COUNT_IDX] = selection_kind.count / cap


def encode_batch_screen(batch_state: list[GameState], device: torch.device) -> torch.Tensor:
    batch_size = len(batch_state)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_SCREEN), dtype=np.float32)

    for b, state in enumerate(batch_state):
        _encode_screen_into(state, np_out[b])

    return torch.from_numpy(np_out).to(device)
