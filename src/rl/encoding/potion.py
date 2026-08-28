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

from src.rl.constants import MAX_EVENT_ROLL_POTIONS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import clip_to_slice
from src.rl.types import make_channel_allocator


# Order = fill order = Core's global-offset order
SLICE_POTIONS = [
    Slice(SliceKind.POTION_OWNED, MAX_POTION_SLOTS),
    Slice(SliceKind.POTION_REWARD, MAX_POTION_REWARDS),
    Slice(SliceKind.POTION_SHOP, MAX_SHOP_POTIONS),
    Slice(SliceKind.POTION_EVENT, MAX_EVENT_ROLL_POTIONS),
]
NUM_POTION_TOKENS = sum(slice_.size for slice_ in SLICE_POTIONS)
SLICE_KIND_POTIONS = {slice_.kind for slice_ in SLICE_POTIONS}


_POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


class PotionChannel(StrEnum):
    NAME = auto()
    RARITY = auto()
    REQUIRES_TARGET = auto()
    COMBAT_ONLY = auto()


CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
OFF_POTION_NAME = _alloc(PotionChannel.NAME, len(_POTION_NAME_TO_IDX))
OFF_POTION_RARITY = _alloc(PotionChannel.RARITY, len(_POTION_RARITY_TO_IDX))
OFF_POTION_REQUIRES_TARGET = _alloc(PotionChannel.REQUIRES_TARGET)
OFF_POTION_COMBAT_ONLY = _alloc(PotionChannel.COMBAT_ONLY)

ENCODING_DIM_POTION = sum(w for _, w in CHANNEL_LAYOUT_POTION)

# Per-potion encoding cache keyed by name (rarity/target/effects are fixed by name)
_POTION_ROW_CACHE: dict[PotionName, np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[OFF_POTION_NAME + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[OFF_POTION_RARITY + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full potion encoding into `out`, cached by name (its fields are fixed by name)."""
    row = _POTION_ROW_CACHE.get(potion.name)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[potion.name] = row
    out[:] = row


def _get_potion_entities(kind: SliceKind, game_state: GameState) -> list:
    match kind:
        case SliceKind.POTION_OWNED:
            return game_state.potions
        case SliceKind.POTION_REWARD:
            reward = game_state.reward
            return reward.potions if reward is not None else []
        case SliceKind.POTION_SHOP:
            return game_state.shop.potions if game_state.shop is not None else []
        case SliceKind.POTION_EVENT:
            event = game_state.event
            if event is not None and not event.consumed:  # post-choice: the offer is dead
                return event.roll_potions
            return []
        case _:
            raise ValueError(f"not a potion slice: {kind}")


def encode_batch_potions(
    batch_game_state: list[GameState], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all potion slices into one (B, NUM_POTION_TOKENS, dim) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_POTION_TOKENS, ENCODING_DIM_POTION), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_POTION_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        offset = 0
        for slice_ in SLICE_POTIONS:
            potions = clip_to_slice(_get_potion_entities(slice_.kind, game_state), slice_)
            for i, potion in enumerate(potions):
                if potion is None:  # belt slots are positional; empty = hole
                    continue

                encode_potion_into_w_cache(potion, np_out[b, offset + i])
                np_pad[b, offset + i] = True

            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, effects) for every potion token with effects (belt holes skipped)."""
    offset = 0
    for slice_ in SLICE_POTIONS:
        potions = clip_to_slice(_get_potion_entities(slice_.kind, game_state), slice_)
        for i, potion in enumerate(potions):
            effects = potion.effects if potion is not None else None
            if effects:
                yield offset + i, effects
        offset += slice_.size
