import numpy as np
from slai import GameEnv
from slai import GameState
from slai import Relic
from slai import RelicName
from slai import RelicTier
from slai import get_relic_templates

from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.shared import encode_shared_price_into
from src.rl.encoding.shared import encode_shared_tag_into
from src.rl.encoding.shared import get_record_key_price
from src.rl.tokens import get_character
from src.rl.types import RecordKeyRelic
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
_MAX_POCKETWATCH_PLAYS = GameEnv.MAX_POCKETWATCH_PLAYS


_CHANNEL_LAYOUT_RELIC, _alloc = make_channel_allocator()
_OFF_RELIC_NAME = _alloc("name", len(_IDX_OF_RELIC_NAME))
_OFF_RELIC_TIER = _alloc("tier", len(_IDX_OF_RELIC_TIER))
_OFF_RELIC_USED_UP = _alloc("used_up")
_OFF_RELIC_COUNTER = _alloc("counter")
_OFF_RELIC_COUNTER_PROGRESS = _alloc("counter_progress")
_OFF_RELIC_COUNTER_REMAINING = _alloc("counter_remaining", _COUNTER_MAX)
_OFF_RELIC_LINKED = _alloc("linked")
_OFF_RELIC_ARMED = _alloc("armed")

ENCODING_DIM_RELIC = sum(w for _, w in _CHANNEL_LAYOUT_RELIC)

# Most nonzero columns a Relic row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_RELIC = len(_CHANNEL_LAYOUT_RELIC)

# Each tagged Relic's Tags, by name
_TAGS_OF_RELIC_NAME: dict[RelicName, tuple[Tag, ...]] = {
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


def encode_relic_into(
    relic: Relic, linked: bool, armed: bool, gold: int, np_own: np.ndarray, np_shared: np.ndarray
) -> None:
    """Write a Relic's token: its own columns, then its price and its Tags.

    Every input besides the Relic is in the Relic's record key (`get_record_key_relic`), so a new
    input goes in the key too.

    Parameters
    ----------
    relic : Relic
        The Relic.
    linked : bool
        Whether the Sapphire Key is linked to it.
    armed : bool
        Whether what an owned Relic waits for holds now (is_relic_armed); False for a Relic
        not owned.
    gold : int
        The Character's gold, which its price is read against.
    np_own : np.ndarray
        (ENCODING_DIM_RELIC,) float32. The Relic's own range of its token row, written in place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    np_own[_OFF_RELIC_NAME + _IDX_OF_RELIC_NAME[relic.name]] = 1.0
    np_own[_OFF_RELIC_TIER + _IDX_OF_RELIC_TIER[relic.tier]] = 1.0
    np_own[_OFF_RELIC_USED_UP] = float(relic.used_up)
    np_own[_OFF_RELIC_COUNTER] = min(relic.counter, _COUNTER_MAX) / _COUNTER_MAX

    counter_reset = relic.counter_reset
    if counter_reset > 0:
        # Stone Calendar keeps counting the turns past its threshold
        np_own[_OFF_RELIC_COUNTER_PROGRESS] = min(relic.counter / counter_reset, 1.0)

        # None from the threshold on: Horn Cleat and Captain's Wheel hold there once they fire
        if relic.counter < counter_reset:
            np_own[_OFF_RELIC_COUNTER_REMAINING + counter_reset - relic.counter - 1] = 1.0

    np_own[_OFF_RELIC_LINKED] = float(linked)
    np_own[_OFF_RELIC_ARMED] = float(armed)

    # Shop price
    encode_shared_price_into(relic.price, gold, np_shared)

    # Tags. Shared w/ Cards and Potions
    encode_shared_tag_into(_TAGS_OF_RELIC_NAME.get(relic.name, ()), np_shared)


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
        case RelicName.MeatOnTheBone:
            character = get_character(game_state)

            return character.health * 2 <= character.health_max
        
        case RelicName.Orichalcum:
            return get_character(game_state).block == 0
        
        case RelicName.Pocketwatch:
            return relic.counter <= _MAX_POCKETWATCH_PLAYS
        
        case RelicName.StoneCalendar:
            combat = game_state.combat

            return (
                combat is not None
                and relic.counter == relic.counter_reset
                and not combat.turn_end_started
            )

    return _COUNTER_ARMED_OF_RELIC_NAME.get(relic.name) == relic.counter


def get_record_key_relic(relic: Relic, linked: bool, armed: bool, gold: int) -> RecordKeyRelic:
    """Make a Relic's record key: its identity_hash and everything else its row reads.

    Parameters
    ----------
    relic : Relic
        The Relic.
    linked : bool
        Whether the Sapphire Key is linked to it.
    armed : bool
        Whether what an owned Relic waits for holds now (is_relic_armed).
    gold : int
        The Character's gold, which its price is read against.

    Returns
    -------
    RecordKeyRelic
        Its identity_hash, whether the Sapphire Key is linked to it, whether it is armed, and the
        gold when it has a price.
    """
    return (relic.identity_hash, linked, armed, get_record_key_price(relic.price, gold))
