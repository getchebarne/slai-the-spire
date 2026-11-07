"""Feature layout and static catalogs for card rows."""

from typing import get_args

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

from src.rl.schema.effect import catalog_effects
from src.rl.schema.layout import make_channel_allocator
from src.rl.types import Tag


# Enum maps, keyed by int(): the declaration-order discriminant
MAP_CARD_NAME = {card_name: int(card_name) for card_name in CardName.members()}
MAP_CARD_KIND = {card_kind: int(card_kind) for card_kind in CardKind.members()}
MAP_CARD_COLOR = {card_color: int(card_color) for card_color in CardColor.members()}
MAP_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in CardRarity.members()}
MAP_COST_SCOPE = {scope: int(scope) for scope in CostScope.members()}
MAP_PLAY_SOURCE = {play_source: int(play_source) for play_source in PlaySource.members()}
MAP_CARD_COST_KIND = {cls: i for i, cls in enumerate(get_args(CardCostKind))}

# Cost, derived: the catalog's widest base cost or override amount
COST_MIN = 0
COST_MAX = max(
    max(card_template.cost_base for card_template in get_card_templates()),
    max(
        effect.amount for effect in catalog_effects() if isinstance(effect, EffectSetCostOverride)
    ),
)
_COST_VECTOR_DIM = COST_MAX - COST_MIN + 1
COST_OFFSET_MAX = max(
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


CHANNEL_LAYOUT_CARD, _alloc = make_channel_allocator()
OFF_CARD_NAME = _alloc("name", len(MAP_CARD_NAME))
OFF_CARD_KIND = _alloc("kind", len(MAP_CARD_KIND))
OFF_CARD_COLOR = _alloc("color", len(MAP_CARD_COLOR))
OFF_CARD_RARITY = _alloc("rarity", len(MAP_CARD_RARITY))
OFF_CARD_COST_KIND = _alloc("cost_kind", len(MAP_CARD_COST_KIND))
OFF_CARD_COST_SCOPE = _alloc("cost_scope", len(MAP_COST_SCOPE))  # All-zero = no override
OFF_CARD_COST_VECTOR = _alloc("cost_vector", _COST_VECTOR_DIM)
OFF_CARD_COST_SCALAR = _alloc("cost_scalar")
OFF_CARD_UPGRADED = _alloc("upgraded")
OFF_CARD_EXHAUST = _alloc("exhaust")
OFF_CARD_INNATE = _alloc("innate")
OFF_CARD_ETHEREAL = _alloc("ethereal")
OFF_CARD_RETAIN = _alloc("retain")
OFF_CARD_REQUIRES_TARGET = _alloc("requires_target")
OFF_CARD_PLAYABLE = _alloc("playable")
OFF_CARD_ENERGY_DELTA = _alloc("energy_delta")  # Signed: energy_current - cost
OFF_CARD_COST_BASE = _alloc("cost_base")
OFF_CARD_BOTTLED = _alloc("bottled")
OFF_CARD_COST_OFFSET = _alloc("cost_offset")  # X-Cost offset
OFF_CARD_FREE_TO_PLAY_ONCE = _alloc(
    "free_to_play_once"
)  # Setup, Forethought: the 0 in `cost` lasts one play
OFF_CARD_PLAY_SOURCE = _alloc(
    "play_source", len(MAP_PLAY_SOURCE)
)  # All-zero = not in the play queue

ENCODING_DIM_CARD = sum(w for _, w in CHANNEL_LAYOUT_CARD)

# Most nonzero columns a Card row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_CARD = len(CHANNEL_LAYOUT_CARD)

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
