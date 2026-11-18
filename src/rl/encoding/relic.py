import numpy as np
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


_IDX_OF_RELIC_NAME = {relic_name: int(relic_name) for relic_name in RelicName.members()}
_IDX_OF_RELIC_TIER = {tier: int(tier) for tier in RelicTier.members()}
_COUNTER_MAX = max(template.counter_reset for template in get_relic_templates())

# The counter at which a counting Relic is armed
_COUNTER_ARMED_OF_RELIC_NAME = {
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


_CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
_OFF_RELIC_NAME = _alloc("name", len(_IDX_OF_RELIC_NAME))
_OFF_RELIC_TIER = _alloc("tier", len(_IDX_OF_RELIC_TIER))
_OFF_RELIC_USED_UP = _alloc("used_up")
_OFF_RELIC_COUNTER = _alloc("counter")
_OFF_RELIC_COUNTER_PROGRESS = _alloc(
    "counter_progress"
)  # Counter / counter_reset for firing counters, else 0
_OFF_RELIC_COUNTER_REMAINING = _alloc(
    "counter_remaining", _COUNTER_MAX
)  # Triggers left until it fires, bucket 0 the next one
_OFF_RELIC_LINKED = _alloc("linked")  # The Relic the Sapphire Key is linked to
_OFF_RELIC_ARMED = _alloc("armed")  # Owned: what it waits for holds now (is_relic_armed)

ENCODING_DIM_RELIC = sum(w for _, w in _CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a Relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(_CHANNEL_LAYOUT_RELIC)

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

# Per-relic encoding cache keyed by (name, used_up, counter, linked, armed): all that can change
_RELIC_ROW_CACHE: dict[tuple, np.ndarray] = {}
_RELIC_ROW_CACHE_MAX = 100_000


def encode_relic_into(relic: Relic, linked: bool, armed: bool, np_out: np.ndarray) -> None:
    np_out[_OFF_RELIC_NAME + _IDX_OF_RELIC_NAME[relic.name]] = 1.0
    np_out[_OFF_RELIC_TIER + _IDX_OF_RELIC_TIER[relic.tier]] = 1.0
    np_out[_OFF_RELIC_USED_UP] = float(relic.used_up)
    np_out[_OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        # Stone Calendar keeps counting the turns past its threshold
        np_out[_OFF_RELIC_COUNTER_PROGRESS] = min(relic.counter / counter_reset, 1.0)

        # None from the threshold on: Horn Cleat and Captain's Wheel hold there once they fire
        if relic.counter < counter_reset:
            np_out[_OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0

    np_out[_OFF_RELIC_LINKED] = float(linked)
    np_out[_OFF_RELIC_ARMED] = float(armed)


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
    return _COUNTER_ARMED_OF_RELIC_NAME.get(relic.name) == relic.counter


def encode_relic_into_w_cache(
    relic: Relic, relic_linked: Relic | None, armed: bool, np_out: np.ndarray
) -> None:
    """Write the Relic encoding into `np_out`, cached by (name, used_up, counter, linked, armed).

    Parameters
    ----------
    relic : Relic
        The Relic to encode.
    relic_linked : Relic | None
        The Relic the Sapphire Key is linked to, None without the key on offer.
    armed : bool
        Whether what an owned Relic waits for holds now (is_relic_armed); False for a Relic
        not owned.
    np_out : np.ndarray
        (ENCODING_DIM_RELIC,) float32. The Relic's row, overwritten.
    """
    linked = relic_linked is not None and relic.id == relic_linked.id
    cache_key = (relic.name, relic.used_up, relic.counter, linked, armed)
    np_row = _RELIC_ROW_CACHE.get(cache_key)
    if np_row is None:
        np_row = np.zeros(ENCODING_DIM_RELIC, dtype=np.float32)
        encode_relic_into(relic, linked, armed, np_row)
        np_row.flags.writeable = False
        if len(_RELIC_ROW_CACHE) >= _RELIC_ROW_CACHE_MAX:
            _RELIC_ROW_CACHE.clear()
        _RELIC_ROW_CACHE[cache_key] = np_row
    np_out[:] = np_row
