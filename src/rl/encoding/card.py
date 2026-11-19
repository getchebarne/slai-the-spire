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

from src.rl.constants import MAX_SIZE_CARD_PILE_DRAW
from src.rl.encoding.effect import catalog_effects
from src.rl.encoding.layout import make_channel_allocator
from src.rl.types import Tag


# Enum maps, keyed by int(): the declaration-order discriminant
_IDX_OF_CARD_NAME = {card_name: int(card_name) for card_name in CardName.members()}
_IDX_OF_CARD_KIND = {card_kind: int(card_kind) for card_kind in CardKind.members()}
_IDX_OF_CARD_COLOR = {card_color: int(card_color) for card_color in CardColor.members()}
_IDX_OF_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in CardRarity.members()}
_IDX_OF_COST_SCOPE = {scope: int(scope) for scope in CostScope.members()}
_IDX_OF_PLAY_SOURCE = {play_source: int(play_source) for play_source in PlaySource.members()}
_IDX_OF_CARD_COST_KIND = {cls: idx_class for idx_class, cls in enumerate(get_args(CardCostKind))}

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


# Per-card identity + energy + draw depth


_CHANNEL_LAYOUT_CARD, _alloc = make_channel_allocator()
_OFF_CARD_NAME = _alloc("name", len(_IDX_OF_CARD_NAME))
_OFF_CARD_KIND = _alloc("kind", len(_IDX_OF_CARD_KIND))
_OFF_CARD_COLOR = _alloc("color", len(_IDX_OF_CARD_COLOR))
_OFF_CARD_RARITY = _alloc("rarity", len(_IDX_OF_CARD_RARITY))
_OFF_CARD_COST_KIND = _alloc("cost_kind", len(_IDX_OF_CARD_COST_KIND))
_OFF_CARD_COST_SCOPE = _alloc("cost_scope", len(_IDX_OF_COST_SCOPE))  # All-zero = no override
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
    "play_source", len(_IDX_OF_PLAY_SOURCE)
)  # All-zero = not in the play queue
_OFF_CARD_DRAW_DEPTH = _alloc(
    "draw_depth", MAX_SIZE_CARD_PILE_DRAW
)  # The Cards above it in the draw pile. All-zero = not visible
_OFF_CARD_DRAW_DEPTH_SCALAR = _alloc(
    "draw_depth_scalar"
)  # The same count over MAX_SIZE_CARD_PILE_DRAW, at most 1
_OFF_CARD_DRAW_BELOW_SCALAR = _alloc(
    "draw_below_scalar"
)  # The Cards below it in the draw pile, likewise

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


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def encode_card_into(
    card: Card, energy_current: int, draw_below: int | None, np_out: np.ndarray
) -> None:
    np_out[_OFF_CARD_NAME + _IDX_OF_CARD_NAME[card.name]] = 1.0
    np_out[_OFF_CARD_KIND + _IDX_OF_CARD_KIND[card.kind]] = 1.0
    np_out[_OFF_CARD_COLOR + _IDX_OF_CARD_COLOR[card.color]] = 1.0
    np_out[_OFF_CARD_RARITY + _IDX_OF_CARD_RARITY[card.rarity]] = 1.0
    np_out[_OFF_CARD_COST_KIND + _IDX_OF_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    if card.cost_override_scope is not None:
        np_out[_OFF_CARD_COST_SCOPE + _IDX_OF_COST_SCOPE[card.cost_override_scope]] = 1.0

    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    np_out[_OFF_CARD_COST_VECTOR + cost] = 1.0

    np_out[_OFF_CARD_COST_SCALAR] = cost / _COST_MAX
    np_out[_OFF_CARD_UPGRADED] = float(card.upgraded)
    np_out[_OFF_CARD_EXHAUST] = float(card.exhaust)
    np_out[_OFF_CARD_INNATE] = float(card.innate)
    np_out[_OFF_CARD_ETHEREAL] = float(card.ethereal)
    np_out[_OFF_CARD_RETAIN] = float(card.retain)
    np_out[_OFF_CARD_REQUIRES_TARGET] = float(card.requires_target)
    np_out[_OFF_CARD_PLAYABLE] = float(card.playable)
    np_out[_OFF_CARD_ENERGY_DELTA] = (
        max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    )
    np_out[_OFF_CARD_COST_BASE] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    np_out[_OFF_CARD_BOTTLED] = float(card.bottled)
    np_out[_OFF_CARD_COST_OFFSET] = _cost_offset(card) / _COST_OFFSET_MAX
    np_out[_OFF_CARD_FREE_TO_PLAY_ONCE] = float(card.free_to_play_once)
    if card.play_source is not None:
        np_out[_OFF_CARD_PLAY_SOURCE + _IDX_OF_PLAY_SOURCE[card.play_source]] = 1.0
    if card.draw_depth is not None:
        np_out[_OFF_CARD_DRAW_DEPTH + min(card.draw_depth, MAX_SIZE_CARD_PILE_DRAW - 1)] = 1.0
        np_out[_OFF_CARD_DRAW_DEPTH_SCALAR] = (
            min(card.draw_depth, MAX_SIZE_CARD_PILE_DRAW) / MAX_SIZE_CARD_PILE_DRAW
        )
    if draw_below is not None:
        np_out[_OFF_CARD_DRAW_BELOW_SCALAR] = (
            min(draw_below, MAX_SIZE_CARD_PILE_DRAW) / MAX_SIZE_CARD_PILE_DRAW
        )
