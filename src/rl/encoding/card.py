from typing import get_args

import numpy as np
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardCostKindXCost
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import CostScope
from slai import EffectSetCostOverride
from slai import PlaySource
from slai import get_card_templates

from src.rl.encoding.effect_layout import catalog_effects
from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


# Enum maps, keyed by int(): the declaration-order discriminant
_MAP_CARD_NAME = {card_name: int(card_name) for card_name in CardName.members()}
_MAP_CARD_KIND = {card_kind: int(card_kind) for card_kind in CardKind.members()}
_MAP_CARD_COLOR = {card_color: int(card_color) for card_color in CardColor.members()}
_MAP_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in CardRarity.members()}
_MAP_COST_SCOPE = {scope: int(scope) for scope in CostScope.members()}
_MAP_PLAY_SOURCE = {play_source: int(play_source) for play_source in PlaySource.members()}
_MAP_CARD_COST_KIND = {cls: i for i, cls in enumerate(get_args(CardCostKind))}

# Cost, derived: the catalog's widest base cost or override amount
_COST_MIN = 0
_COST_MAX = max(
    max(card_template.cost_base for card_template in get_card_templates()),
    max(
        effect.amount for effect in catalog_effects() if isinstance(effect, EffectSetCostOverride)
    ),
)
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1
_COST_OFFSET_MAX = max(
    1.0,
    max(
        (
            abs(card_template.cost_kind.offset)
            for card_template in get_card_templates()
            if isinstance(card_template.cost_kind, CardCostKindXCost)
        ),
        default=0,
    ),
)


# Per-card identity + energy


_CHANNEL_LAYOUT_CARD, _alloc = make_channel_allocator()
_OFF_CARD_NAME = _alloc("name", len(_MAP_CARD_NAME))
_OFF_CARD_KIND = _alloc("kind", len(_MAP_CARD_KIND))
_OFF_CARD_COLOR = _alloc("color", len(_MAP_CARD_COLOR))
_OFF_CARD_RARITY = _alloc("rarity", len(_MAP_CARD_RARITY))
_OFF_CARD_COST_KIND = _alloc("cost_kind", len(_MAP_CARD_COST_KIND))
_OFF_CARD_COST_SCOPE = _alloc("cost_scope", len(_MAP_COST_SCOPE))  # All-zero = no override
_OFF_CARD_COST_VECTOR = _alloc("cost_vector", _COST_VECTOR_DIM)
_OFF_CARD_COST_SCALAR = _alloc("cost_scalar")
_OFF_CARD_UPGRADED = _alloc("upgraded")
_OFF_CARD_EXHAUST = _alloc("exhaust")
_OFF_CARD_INNATE = _alloc("innate")
_OFF_CARD_ETHEREAL = _alloc("ethereal")
_OFF_CARD_RETAIN = _alloc("retain")
_OFF_CARD_REQUIRES_TARGET = _alloc("requires_target")
_OFF_CARD_PLAYABLE = _alloc("playable")
_OFF_CARD_ENERGY_DELTA = _alloc("energy_delta")  # Signed: energy_current - cost
_OFF_CARD_COST_BASE = _alloc("cost_base")
_OFF_CARD_BOTTLED = _alloc("bottled")
_OFF_CARD_COST_OFFSET = _alloc("cost_offset")  # X-Cost offset
_OFF_CARD_FREE_TO_PLAY_ONCE = _alloc(
    "free_to_play_once"
)  # Setup, Forethought: the 0 in `cost` lasts one play
_OFF_CARD_PLAY_SOURCE = _alloc(
    "play_source", len(_MAP_PLAY_SOURCE)
)  # All-zero = not in the play queue

ENCODING_DIM_CARD = sum(w for _, w in _CHANNEL_LAYOUT_CARD)

# Most nonzero columns a Card row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_CARD = len(_CHANNEL_LAYOUT_CARD)

# Each tagged Card's Tags, by name: an upgraded Card keeps them
TAGS_OF_CARD_NAME: dict[CardName, tuple[Tag, ...]] = {
    CardName.Accuracy: (Tag.NINJA,),
    CardName.Acrobatics: (Tag.DISCARD,),
    CardName.AfterImage: (Tag.NINJA,),
    CardName.AllOutAttack: (Tag.DISCARD,),
    CardName.AThousandCuts: (Tag.NINJA,),
    CardName.Backstab: (Tag.NINJA,),
    CardName.Bane: (Tag.POISON,),
    CardName.BladeDance: (Tag.NINJA,),
    CardName.BouncingFlask: (Tag.POISON,),
    CardName.Burst: (Tag.SETUP,),
    CardName.CalculatedGamble: (Tag.DISCARD,),
    CardName.Catalyst: (Tag.POISON, Tag.SETUP),
    CardName.Choke: (Tag.NINJA,),
    CardName.CloakAndDagger: (Tag.NINJA,),
    CardName.Concentrate: (Tag.DISCARD,),
    CardName.CorpseExplosion: (Tag.POISON,),
    CardName.CripplingPoison: (Tag.POISON,),
    CardName.DaggerThrow: (Tag.DISCARD,),
    CardName.DeadlyPoison: (Tag.POISON,),
    CardName.Doppelganger: (Tag.SETUP,),
    CardName.EndlessAgony: (Tag.NINJA,),
    CardName.Envenom: (Tag.POISON, Tag.NINJA),
    CardName.Eviscerate: (Tag.DISCARD,),
    CardName.Finisher: (Tag.NINJA,),
    CardName.FlashOfSteel: (Tag.NINJA,),
    CardName.Forethought: (Tag.SETUP,),
    CardName.GrandFinale: (Tag.SETUP,),
    CardName.InfiniteBlades: (Tag.NINJA,),
    CardName.Madness: (Tag.SETUP,),
    CardName.Neutralize: (Tag.NINJA,),
    CardName.Nightmare: (Tag.SETUP,),
    CardName.NoxiousFumes: (Tag.POISON,),
    CardName.Outmaneuver: (Tag.SETUP,),
    CardName.Panache: (Tag.NINJA,),
    CardName.PoisonedStab: (Tag.POISON,),
    CardName.Prepared: (Tag.DISCARD,),
    CardName.Reflex: (Tag.DISCARD,),
    CardName.Setup: (Tag.SETUP,),
    CardName.Shiv: (Tag.NINJA,),
    CardName.SneakyStrike: (Tag.DISCARD,),
    CardName.StormOfSteel: (Tag.NINJA,),
    CardName.Survivor: (Tag.DISCARD,),
    CardName.Tactician: (Tag.DISCARD,),
    CardName.ThinkingAhead: (Tag.SETUP,),
    CardName.ToolsOfTheTrade: (Tag.DISCARD,),
    CardName.Unload: (Tag.DISCARD,),
    CardName.WellLaidPlans: (Tag.SETUP,),
}

# Per-card encoding cache, keyed by (card_key, energy) — see card_key
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def _encode_card_into(card: Card, energy_current: int, out: np.ndarray) -> None:
    out[_OFF_CARD_NAME + _MAP_CARD_NAME[card.name]] = 1.0
    out[_OFF_CARD_KIND + _MAP_CARD_KIND[card.kind]] = 1.0
    out[_OFF_CARD_COLOR + _MAP_CARD_COLOR[card.color]] = 1.0
    out[_OFF_CARD_RARITY + _MAP_CARD_RARITY[card.rarity]] = 1.0
    out[_OFF_CARD_COST_KIND + _MAP_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    if card.cost_override_scope is not None:
        out[_OFF_CARD_COST_SCOPE + _MAP_COST_SCOPE[card.cost_override_scope]] = 1.0

    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    out[_OFF_CARD_COST_VECTOR + cost] = 1.0

    out[_OFF_CARD_COST_SCALAR] = cost / _COST_MAX
    out[_OFF_CARD_UPGRADED] = float(card.upgraded)
    out[_OFF_CARD_EXHAUST] = float(card.exhaust)
    out[_OFF_CARD_INNATE] = float(card.innate)
    out[_OFF_CARD_ETHEREAL] = float(card.ethereal)
    out[_OFF_CARD_RETAIN] = float(card.retain)
    out[_OFF_CARD_REQUIRES_TARGET] = float(card.requires_target)
    out[_OFF_CARD_PLAYABLE] = float(card.playable)
    out[_OFF_CARD_ENERGY_DELTA] = (
        max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    )
    out[_OFF_CARD_COST_BASE] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[_OFF_CARD_BOTTLED] = float(card.bottled)
    out[_OFF_CARD_COST_OFFSET] = _cost_offset(card) / _COST_OFFSET_MAX
    out[_OFF_CARD_FREE_TO_PLAY_ONCE] = float(card.free_to_play_once)
    if card.play_source is not None:
        out[_OFF_CARD_PLAY_SOURCE + _MAP_PLAY_SOURCE[card.play_source]] = 1.0


def card_key(card: Card) -> tuple:
    """Return the fields _encode_card_into reads, minus energy.

    Two Cards with equal identity encode the same ROW at any given energy. It keys the row cache
    and, with effect_key, masks' identity dedup; Effects are rows of their own, so absent.

    Explicit rather than hash(Card): live Entities carry a unique arena `id` that participates in
    the engine's __hash__, so hashing the object would make every Card distinct and silently kill
    the cache.

    Parameters
    ----------
    card : Card
        The Card to key.

    Returns
    -------
    tuple
        The Card's identity key: those fields' values.
    """

    return (
        card.name,
        card.kind,
        card.color,
        card.rarity,
        type(card.cost_kind),
        _cost_offset(card),
        card.cost,
        card.cost_base,
        card.cost_override_scope,
        card.upgraded,
        card.exhaust,
        card.innate,
        card.ethereal,
        card.retain,
        card.requires_target,
        card.playable,
        card.bottled,
        card.free_to_play_once,
        card.play_source,
    )


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    # Try cache hit
    cache_key = (card_key(card), energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)

    # Cache miss
    if row is None:
        # Encode Card
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        _encode_card_into(card, energy_current, row)

        # Write into cache
        row.flags.writeable = False
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row

    # Write into output array
    out[:] = row
