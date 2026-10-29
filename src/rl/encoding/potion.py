from enum import StrEnum
from enum import auto

import numpy as np
from slai import GameState
from slai import Potion
from slai import PotionName
from slai import PotionRarity

from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SIZE_POTION_EVENT_ROLL
from src.rl.types import TokenKind
from src.rl.types import make_channel_allocator


_POTION_NAME_TO_IDX = {potion_name: int(potion_name) for potion_name in PotionName.members()}
_POTION_RARITY_TO_IDX = {rarity: int(rarity) for rarity in PotionRarity.members()}


class PotionChannel(StrEnum):
    NAME = auto()
    RARITY = auto()
    REQUIRES_TARGET = auto()
    COMBAT_ONLY = auto()
    DOUBLED = auto()  # Sacred Bark: its Effect rows already carry the doubled amounts


CHANNEL_LAYOUT_POTION, _alloc = make_channel_allocator()
OFF_POTION_NAME = _alloc(PotionChannel.NAME, len(_POTION_NAME_TO_IDX))
OFF_POTION_RARITY = _alloc(PotionChannel.RARITY, len(_POTION_RARITY_TO_IDX))
OFF_POTION_REQUIRES_TARGET = _alloc(PotionChannel.REQUIRES_TARGET)
OFF_POTION_COMBAT_ONLY = _alloc(PotionChannel.COMBAT_ONLY)
OFF_POTION_DOUBLED = _alloc(PotionChannel.DOUBLED)

ENCODING_DIM_POTION = sum(w for _, w in CHANNEL_LAYOUT_POTION)

# Most nonzero columns a Potion row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_POTION = len(CHANNEL_LAYOUT_POTION)

# Per-potion encoding cache keyed by (name, doubled): every other field is fixed by name
_POTION_ROW_CACHE: dict[tuple[PotionName, bool], np.ndarray] = {}
_POTION_ROW_CACHE_MAX = 100_000


def encode_potion_into(potion: Potion, out: np.ndarray) -> None:
    out[OFF_POTION_NAME + _POTION_NAME_TO_IDX[potion.name]] = 1.0
    out[OFF_POTION_RARITY + _POTION_RARITY_TO_IDX[potion.rarity]] = 1.0
    out[OFF_POTION_REQUIRES_TARGET] = float(potion.requires_target)
    out[OFF_POTION_COMBAT_ONLY] = float(potion.combat_only)
    out[OFF_POTION_DOUBLED] = float(potion.doubled)


def encode_potion_into_w_cache(potion: Potion, out: np.ndarray) -> None:
    """Write the full Potion encoding into `out`, cached by (name, doubled).

    Parameters
    ----------
    potion : Potion
        The Potion to encode.
    out : np.ndarray
        (ENCODING_DIM_POTION,) float32. The Potion's row, overwritten.
    """
    cache_key = (potion.name, potion.doubled)  # Sacred Bark's variant keeps the name
    row = _POTION_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_POTION, dtype=np.float32)
        encode_potion_into(potion, row)
        row.flags.writeable = False  # Guard the cached master copy
        if len(_POTION_ROW_CACHE) >= _POTION_ROW_CACHE_MAX:
            _POTION_ROW_CACHE.clear()
        _POTION_ROW_CACHE[cache_key] = row
    out[:] = row


def get_potion_entities(token_kind: TokenKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap (all engine-bounded, so never lossy)
    match token_kind:
        case TokenKind.POTION_OWNED:
            return game_state.potions[:MAX_POTION_SLOTS]
        case TokenKind.POTION_REWARD:
            reward = game_state.reward
            return reward.potions[:MAX_POTION_REWARDS] if reward is not None else []
        case TokenKind.POTION_SHOP:
            shop = game_state.shop
            return shop.potions[:MAX_SHOP_POTIONS] if shop is not None else []
        case TokenKind.POTION_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:  # Post-choice: the offer is dead
                return event.potion_event_roll[:MAX_SIZE_POTION_EVENT_ROLL]
            return []
        case _:
            raise ValueError(f"not a potion token kind: {token_kind}")
