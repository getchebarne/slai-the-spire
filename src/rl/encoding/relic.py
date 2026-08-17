import numpy as np
import torch
from slai import EventKindNloth
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
from src.rl.types import clip_to_slice

# Order = fill order = Core's global-offset order
SLICE_RELICS = [
    Slice(SliceKind.RELIC_OWNED, MAX_RELICS),
    Slice(SliceKind.RELIC_REWARD, MAX_RELIC_REWARDS),
    Slice(SliceKind.RELIC_SHOP, MAX_SHOP_RELICS),
    Slice(SliceKind.RELIC_EVENT, 2),  # N'loth's two rolled owned relics, option order
]
SLICE_KIND_RELICS = {slice_.kind for slice_ in SLICE_RELICS}
NUM_RELIC_TOKENS = sum(slice_.size for slice_ in SLICE_RELICS)


# int() keying: declaration-order discriminant, stable across engine bumps
_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in members(RelicName)}
_COUNTER_MAX = 10  # Nunchaku 10, Girya 3, Matryoshka 2, Incense Burner 6, Sundial 3

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
            reward = game_state.reward
            return reward.relics if reward is not None else []
        case SliceKind.RELIC_SHOP:
            return game_state.shop.relics if game_state.shop is not None else []
        case SliceKind.RELIC_EVENT:
            event = game_state.event
            if (
                event is not None
                and not event.consumed  # post-choice aftermath: the offer is dead
                and isinstance(event.kind, EventKindNloth)
            ):
                # The payload names are baked AT SPAWN; after the agent gives a
                # relic away (RelicLose), post-choice snapshots still carry the
                # name with the relic gone from the belt — encode only what is
                # still owned. (Crashed live at ~12M steps as a loud KeyError
                # when this filter was briefly removed as "impossible".)
                by_name = {relic.name: relic for relic in game_state.relics}
                pair = (event.kind.relic_first, event.kind.relic_second)
                return [by_name[name] for name in pair if name in by_name]
            return []
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
            relics = clip_to_slice(_get_relic_entities(slice_.kind, game_state), slice_)
            for i, relic in enumerate(relics):
                encode_relic_into_w_cache(relic, np_out[b, offset + i])
                np_pad[b, offset + i] = True
            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )
