from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Effect
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.types import EffectSource
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator


# Pack order within the potion block; every source is engine-bounded
SLICE_KINDS_POTION = (
    SliceKind.POTION_OWNED,
    SliceKind.POTION_REWARD,
    SliceKind.POTION_SHOP,
    SliceKind.POTION_EVENT_ROLL,
)


_POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


class PotionChannel(StrEnum):
    NAME = auto()
    RARITY = auto()
    REQUIRES_TARGET = auto()
    COMBAT_ONLY = auto()
    DOUBLED = auto()  # Sacred Bark: its effect rows already carry the doubled amounts


CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
OFF_POTION_NAME = _alloc(PotionChannel.NAME, len(_POTION_NAME_TO_IDX))
OFF_POTION_RARITY = _alloc(PotionChannel.RARITY, len(_POTION_RARITY_TO_IDX))
OFF_POTION_REQUIRES_TARGET = _alloc(PotionChannel.REQUIRES_TARGET)
OFF_POTION_COMBAT_ONLY = _alloc(PotionChannel.COMBAT_ONLY)
OFF_POTION_DOUBLED = _alloc(PotionChannel.DOUBLED)

ENCODING_DIM_POTION = sum(w for _, w in CHANNEL_LAYOUT_POTION)

# Per-potion encoding cache keyed by name (rarity/target/effects are fixed by name)
_POTION_ROW_CACHE: dict[tuple[PotionName, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[OFF_POTION_NAME + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[OFF_POTION_RARITY + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    out[OFF_POTION_DOUBLED] = float(potion.doubled)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full potion encoding into `out`, cached by name (its fields are fixed by name)."""
    cache_key = (potion.name, potion.doubled)  # Sacred Bark's variant keeps the name
    row = _POTION_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_potion_entities(kind: SliceKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap (all engine-bounded, so never lossy)
    match kind:
        case SliceKind.POTION_OWNED:
            return game_state.potions[:MAX_POTION_SLOTS]
        case SliceKind.POTION_REWARD:
            reward = game_state.reward
            return reward.potions[:MAX_POTION_REWARDS] if reward is not None else []
        case SliceKind.POTION_SHOP:
            shop = game_state.shop
            return shop.potions[:MAX_SHOP_POTIONS] if shop is not None else []
        case SliceKind.POTION_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:  # post-choice: the offer is dead
                return event.potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL]
            return []
        case _:
            raise ValueError(f"not a potion kind: {kind}")


def iter_packed_potions(game_state: GameState) -> Iterator[tuple[int, SliceKind, Potion]]:
    """(w, kind, potion) for every potion token: the kinds back to back in
    SLICE_KINDS_POTION order, each collection in engine order."""
    w = 0
    for kind in SLICE_KINDS_POTION:
        for potion in _get_potion_entities(kind, game_state):
            yield w, kind, potion
            w += 1


def encode_batch_potions(
    game_states: list[GameState], device: torch.device
) -> tuple[torch.Tensor, np.ndarray, list[list[tuple[int, EffectSource, Effect]]]]:
    """(B, W, ENCODING_DIM_POTION) rows + the (B, W) kind of each token + each state's
    (offset, source, effect) rows, keyed by the same w as the token they belong to;
    W is the batch's longest packed potion sequence."""
    batch_tokens = [list(iter_packed_potions(game_state)) for game_state in game_states]
    width = max((len(tokens) for tokens in batch_tokens), default=0)
    np_out = np.zeros((len(game_states), width, ENCODING_DIM_POTION), dtype=np.float32)
    np_idxs = np.full((len(game_states), width), -1, dtype=np.int16)
    batch_effect_rows: list[list[tuple[int, EffectSource, Effect]]] = []

    for b, tokens in enumerate(batch_tokens):
        effect_rows: list[tuple[int, EffectSource, Effect]] = []
        for w, kind, potion in tokens:
            encode_potion_into_w_cache(potion, np_out[b, w])
            np_idxs[b, w] = kind
            effects = potion.effects  # bound once: each read clones the Vec
            if effects:
                effect_rows += [(w, EffectSource.POTION_USE, effect) for effect in effects]
        batch_effect_rows.append(effect_rows)

    return torch.from_numpy(np_out).to(device), np_idxs, batch_effect_rows
