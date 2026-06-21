import numpy as np
import torch
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import members

from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.types import Slice
from src.rl.types import SliceKind


# Order = fill order = Core's global-offset order
SLICE_RELICS = [
    Slice(SliceKind.RELIC_OWNED, MAX_RELICS),
    Slice(SliceKind.RELIC_REWARD, MAX_RELIC_REWARDS),
    Slice(SliceKind.RELIC_SHOP, MAX_SHOP_RELICS),
]
SLICE_KIND_RELICS = {slice_.kind for slice_ in SLICE_RELICS}
NUM_RELIC_TOKENS = sum(slice_.size for slice_ in SLICE_RELICS)


_RELIC_NAME_TO_IDX = {relic_name: i for i, relic_name in enumerate(members(RelicName))}
_COUNTER_MAX = 9

ENCODING_DIM_RELIC = (
    len(_RELIC_NAME_TO_IDX)  # Name OHE
    + 1  # Used up
    + 1  # Counter scalar
    + ENCODING_DIM_EFFECTS  # Combat-start effect blocks (trigger timing rides on the name)
)

# Per-relic encoding cache keyed by (name, used_up, clamped counter) — the only mutable fields
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


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


def encode_relic_into_w_cache(relic: Relic, out: np.ndarray) -> None:
    """Write the full relic encoding into `out`, cached by (name, used_up, clamped counter)."""
    cache_key = (relic.name, relic.used_up, min(relic.counter, _COUNTER_MAX))
    row = _RELIC_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, 0, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_relic_entities(kind: SliceKind, game_state: GameState) -> list:
    match kind:
        case SliceKind.RELIC_OWNED:
            return game_state.relics
        case SliceKind.RELIC_REWARD:
            return (
                [game_state.reward.relic]
                if game_state.reward is not None and game_state.reward.relic is not None
                else []
            )
        case SliceKind.RELIC_SHOP:
            return game_state.shop.relics if game_state.shop is not None else []
        case _:
            raise ValueError(f"not a relic slice: {kind}")


def encode_batch_relics(
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all relic slices into one (B, NUM_RELIC_TOKENS, ENCODING_DIM_RELIC) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_RELIC_TOKENS, ENCODING_DIM_RELIC), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_RELIC_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        offset = 0
        for slice_ in SLICE_RELICS:
            for i, relic in enumerate(_get_relic_entities(slice_.kind, game_state)):
                encode_relic_into_w_cache(relic, np_out[b, offset + i])
                np_pad[b, offset + i] = True
            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )
