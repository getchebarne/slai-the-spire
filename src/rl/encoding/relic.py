from enum import StrEnum
from enum import auto

import numpy as np
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_RELIC_EVENT_ROLL
from src.rl.types import Tag
from src.rl.types import TokenKind
from src.rl.types import capped
from src.rl.types import make_channel_allocator


_RELIC_NAME_TO_IDX = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_RELIC_TIER_TO_IDX = {tier: int(tier) for tier in RelicTier.members()}
_COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())

# The counter at which a counting Relic is armed
_MAP_RELIC_NAME_COUNTER_ARMED = {
    RelicName.AncientTeaSet: 1,  # Primed at a Rest Site for the next Combat
    RelicName.ArtOfWar: 0,  # No Attack played this turn: 1 energy next turn
    RelicName.CentennialPuzzle: 0,  # The first HP lost this Combat draws 3
    RelicName.HoveringKite: 0,  # The first discard this turn gives 1 energy
    RelicName.Necronomicon: 0,  # The first Attack costing 2 or more this turn plays twice
    RelicName.PenNib: 9,  # The next Attack deals double
    RelicName.TinyChest: 3,  # The next "?" Room is a Treasure Room
}

# Pocketwatch: this many plays or fewer draw 3 next turn
_POCKETWATCH_PLAYS_ARMED = 3


class RelicChannel(StrEnum):
    NAME = auto()
    TIER = auto()
    USED_UP = auto()
    COUNTER = auto()
    COUNTER_PROGRESS = auto()  # Counter / counter_reset for firing counters, else 0
    COUNTER_REMAINING = auto()  # Triggers left until it fires, bucket 0 the next one
    LINKED = auto()  # The Relic the Sapphire Key is linked to
    ARMED = auto()  # Owned: what it waits for holds now (is_relic_armed)


CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
OFF_RELIC_NAME = _alloc(RelicChannel.NAME, len(_RELIC_NAME_TO_IDX))
OFF_RELIC_TIER = _alloc(RelicChannel.TIER, len(_RELIC_TIER_TO_IDX))
OFF_RELIC_USED_UP = _alloc(RelicChannel.USED_UP)
OFF_RELIC_COUNTER = _alloc(RelicChannel.COUNTER)
OFF_RELIC_COUNTER_PROGRESS = _alloc(RelicChannel.COUNTER_PROGRESS)
OFF_RELIC_COUNTER_REMAINING = _alloc(RelicChannel.COUNTER_REMAINING, _COUNTER_MAX)
OFF_RELIC_LINKED = _alloc(RelicChannel.LINKED)
OFF_RELIC_ARMED = _alloc(RelicChannel.ARMED)

ENCODING_DIM_RELIC = sum(w for _, w in CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a Relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(CHANNEL_LAYOUT_RELIC)

# Each tagged Relic's Tags, by name
TAGS_OF_RELIC_NAME: dict[RelicName, tuple[Tag, ...]] = {
    RelicName.GamblingChip: (Tag.DISCARD,),
    RelicName.HoveringKite: (Tag.DISCARD,),
    RelicName.InkBottle: (Tag.NINJA,),
    RelicName.Kunai: (Tag.NINJA,),
    RelicName.NinjaScroll: (Tag.NINJA,),
    RelicName.Nunchaku: (Tag.NINJA,),
    RelicName.OrnamentalFan: (Tag.NINJA,),
    RelicName.PenNib: (Tag.NINJA,),
    RelicName.RunicPyramid: (Tag.SETUP,),
    RelicName.Shuriken: (Tag.NINJA,),
    RelicName.SneckoSkull: (Tag.POISON,),
    RelicName.TheSpecimen: (Tag.POISON,),
    RelicName.Tingsha: (Tag.DISCARD,),
    RelicName.ToughBandages: (Tag.DISCARD,),
    RelicName.WristBlade: (Tag.NINJA,),
}

# Per-relic encoding cache keyed by (name, used_up, counter) — the only mutable fields
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, out: np.ndarray) -> None:
    out[OFF_RELIC_NAME + _RELIC_NAME_TO_IDX[relic.name]] = 1.0
    out[OFF_RELIC_TIER + _RELIC_TIER_TO_IDX[relic.tier]] = 1.0
    out[OFF_RELIC_USED_UP] = float(relic.used_up)
    out[OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        # Stone Calendar keeps counting the turns past its threshold
        out[OFF_RELIC_COUNTER_PROGRESS] = min(relic.counter / counter_reset, 1.0)

        # None from the threshold on: Horn Cleat and Captain's Wheel hold there once they fire
        if relic.counter < counter_reset:
            out[OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0


def is_relic_armed(relic: Relic, game_state: GameState) -> bool:
    """Return whether what an owned Relic waits for holds now.

    Parameters
    ----------
    relic : Relic
        The owned Relic.
    game_state : GameState
        The state, whose Character and Combat some Relics read.

    Returns
    -------
    bool
        True when the Relic's condition holds; False for a Relic without one.
    """
    match relic.name:
        case RelicName.MeatOnTheBone:  # Ending a Combat at half HP or less heals 12
            character = game_state.character
            return character.health * 2 <= character.health_max
        case RelicName.Orichalcum:  # Ending the turn with no block gains 6
            return game_state.character.block == 0
        case RelicName.Pocketwatch:
            return relic.counter <= _POCKETWATCH_PLAYS_ARMED
        case RelicName.StoneCalendar:  # This turn's end deals 52 to every Monster
            combat = game_state.combat
            return (
                combat is not None
                and relic.counter == relic.counter_reset
                and not combat.turn_end_started
            )
    return _MAP_RELIC_NAME_COUNTER_ARMED.get(relic.name) == relic.counter


def encode_relic_into_w_cache(relic: Relic, out: np.ndarray) -> None:
    """Write the Relic encoding into `out`, cached by (name, used_up, counter).

    LINKED and ARMED read more than the Relic, so they stay out of the cached row.

    Parameters
    ----------
    relic : Relic
        The Relic to encode.
    out : np.ndarray
        (ENCODING_DIM_RELIC,) float32. The Relic's row, overwritten.
    """
    cache_key = (relic.name, relic.used_up, relic.counter)
    row = _RELIC_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, row)
        row.flags.writeable = False
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = row
    out[:] = row


def get_relic_entities(token_kind: TokenKind, game_state: GameState) -> list:
    # Each collection truncated to its own cap: a Relic past it has no token
    match token_kind:
        case TokenKind.RELIC_OWNED:
            return capped(token_kind, game_state.relics, MAX_RELICS)
        case TokenKind.RELIC_REWARD:
            reward = game_state.reward
            return reward.relics[:MAX_RELIC_REWARDS] if reward is not None else []
        case TokenKind.RELIC_SHOP:
            shop = game_state.shop
            return shop.relics[:MAX_SHOP_RELICS] if shop is not None else []
        case TokenKind.RELIC_EVENT_ROLL:
            event = game_state.event
            return event.relic_event_roll[:MAX_SIZE_RELIC_EVENT_ROLL] if event is not None else []
        case _:
            raise ValueError(f"not a relic token kind: {token_kind}")
