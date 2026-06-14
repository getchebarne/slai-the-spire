"""Entity layout registry — the single source of truth for entity segments.

Every entity the model sees lives in exactly one segment (a card in hand, a relic
in the shop, a monster). Segments of one entity class share an encoding dim and a
projection, so the encoding layer emits ONE tensor per class (segments are its
class-local slices, CLASS_SLICE) and the projector concatenates the projected
class tensors into ONE token tensor (segments are its global slices, GLOBAL_SLICE).
Registry order is class-grouped, so a class tensor drops into the token tensor as
a contiguous block — no permutation anywhere encode -> project -> transformer.

Consumers (encoding/, models/, action_space/) import the derived artifacts below;
nothing outside this module enumerates segments, sizes, or order.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Callable

import torch
from slai import GameState

from src.rl.constants import MAP_WIDTH
from src.rl.constants import MAX_EVENT_OPTIONS
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MAX_POTION_REWARDS
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_RELIC_REWARDS
from src.rl.constants import MAX_RELICS
from src.rl.constants import MAX_SHOP_POTIONS
from src.rl.constants import MAX_SHOP_RELICS
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.constants import MAX_SIZE_REWARD_CARDS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.types import Pool


class EntityClass(IntEnum):
    """Entity classes: one encoding dim + one shared projection per class."""

    CARD = 0
    RELIC = 1
    POTION = 2
    MONSTER = 3
    EVENT = 4
    CHARACTER = 5


class Segment(IntEnum):
    """One entity segment = one contiguous run of token slots. The value is the
    segment's registry position AND its type-embedding row."""

    # Cards
    HAND = 0
    DRAW = 1
    DISCARD = 2
    EXHAUST = 3
    DECK = 4
    DISCOVER = 5
    REWARD_CARDS = 6
    SHOP_CARDS = 7
    # Relics
    RELICS = 8
    REWARD_RELIC = 9
    SHOP_RELICS = 10
    # Potions
    POTIONS = 11
    REWARD_POTION = 12
    SHOP_POTIONS = 13
    # Single-segment classes
    MONSTERS = 14
    EVENT_OPTIONS = 15
    CHARACTER = 16


@dataclass(frozen=True)
class SegmentSpec:
    segment: Segment
    entity_class: EntityClass
    size: int
    # Entities of this segment in a GameState (None = layout-only segment whose
    # bespoke encoder gathers its own inputs: monsters/event options/character)
    getter: Callable[[GameState], list] | None = None
    # Card segments only: the energy a card's playability features encode against
    # (combat piles use current energy; shop/reward cards aren't played -> 0)
    energy: Callable[[GameState], int] | None = None


# The engine draws from the END of the draw pile, so when over the encoder cap the
# getter keeps the next-to-draw tail (encoders truncate the head).
# Shop/reward/event contexts may be absent from a GameState -> empty segment.
REGISTRY: tuple[SegmentSpec, ...] = (
    # Cards
    SegmentSpec(
        Segment.HAND,
        EntityClass.CARD,
        MAX_SIZE_HAND,
        lambda gs: gs.hand,
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.DRAW,
        EntityClass.CARD,
        MAX_SIZE_DRAW_PILE,
        lambda gs: gs.pile_draw[-MAX_SIZE_DRAW_PILE:],
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.DISCARD,
        EntityClass.CARD,
        MAX_SIZE_DISC_PILE,
        lambda gs: gs.pile_discard,
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.EXHAUST,
        EntityClass.CARD,
        MAX_SIZE_EXHAUST,
        lambda gs: gs.pile_exhaust,
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.DECK,
        EntityClass.CARD,
        MAX_SIZE_DECK,
        lambda gs: gs.deck,
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.DISCOVER,
        EntityClass.CARD,
        MAX_SIZE_DISCOVER,
        lambda gs: gs.discover,
        lambda gs: gs.energy.energy_current,
    ),
    SegmentSpec(
        Segment.REWARD_CARDS,
        EntityClass.CARD,
        MAX_SIZE_REWARD_CARDS,
        lambda gs: gs.reward.cards if gs.reward is not None else [],
        lambda gs: 0,
    ),
    SegmentSpec(
        Segment.SHOP_CARDS,
        EntityClass.CARD,
        MAX_SIZE_SHOP_CARDS,
        lambda gs: gs.shop.cards if gs.shop is not None else [],
        lambda gs: 0,
    ),
    # Relics
    SegmentSpec(
        Segment.RELICS,
        EntityClass.RELIC,
        MAX_RELICS,
        lambda gs: gs.relics,
    ),
    SegmentSpec(
        Segment.REWARD_RELIC,
        EntityClass.RELIC,
        MAX_RELIC_REWARDS,
        lambda gs: [gs.reward.relic] if gs.reward is not None and gs.reward.relic is not None else [],
    ),
    SegmentSpec(
        Segment.SHOP_RELICS,
        EntityClass.RELIC,
        MAX_SHOP_RELICS,
        lambda gs: gs.shop.relics if gs.shop is not None else [],
    ),
    # Potions (belt slots may hold None mid-list; the class encoder skips them in place)
    SegmentSpec(
        Segment.POTIONS,
        EntityClass.POTION,
        MAX_POTION_SLOTS,
        lambda gs: gs.potions,
    ),
    SegmentSpec(
        Segment.REWARD_POTION,
        EntityClass.POTION,
        MAX_POTION_REWARDS,
        lambda gs: [gs.reward.potion] if gs.reward is not None and gs.reward.potion is not None else [],
    ),
    SegmentSpec(
        Segment.SHOP_POTIONS,
        EntityClass.POTION,
        MAX_SHOP_POTIONS,
        lambda gs: gs.shop.potions if gs.shop is not None else [],
    ),
    # Single-segment classes (bespoke encoders: monster needs character
    # health/block and emits incoming damage; event encodes meta + options
    # jointly; character is a flat singleton)
    SegmentSpec(Segment.MONSTERS, EntityClass.MONSTER, MAX_MONSTERS),
    SegmentSpec(Segment.EVENT_OPTIONS, EntityClass.EVENT, MAX_EVENT_OPTIONS),
    SegmentSpec(Segment.CHARACTER, EntityClass.CHARACTER, 1),
)

assert all(
    spec.segment == i for i, spec in enumerate(REGISTRY)
), "REGISTRY order must match Segment values"
assert [spec.entity_class for spec in REGISTRY] == sorted(
    (spec.entity_class for spec in REGISTRY), key=int
), "REGISTRY must be class-grouped (class tensors map to contiguous token blocks)"


# =============================================================================
# Derived artifacts — consumers import these, never re-enumerate segments
# =============================================================================

SPEC: dict[Segment, SegmentSpec] = {spec.segment: spec for spec in REGISTRY}

CLASS_SEGMENTS: dict[EntityClass, list[SegmentSpec]] = {
    entity_class: [spec for spec in REGISTRY if spec.entity_class == entity_class]
    for entity_class in EntityClass
}

# Local slice of each segment within its class tensor (B, sum(class sizes), D)
CLASS_SLICE: dict[Segment, slice] = {}
for _specs in CLASS_SEGMENTS.values():
    _offset = 0
    for _spec in _specs:
        CLASS_SLICE[_spec.segment] = slice(_offset, _offset + _spec.size)
        _offset += _spec.size

# Token count per class tensor
NUM_CLASS_TOKENS: dict[EntityClass, int] = {
    entity_class: sum(spec.size for spec in specs)
    for entity_class, specs in CLASS_SEGMENTS.items()
}

# Global slice of each segment within the concatenated token tensor (B, NUM_TOKENS, D).
# Classes are contiguous blocks in registry order, so global = class offset + local.
GLOBAL_SLICE: dict[Segment, slice] = {}
_offset = 0
for _spec in REGISTRY:
    GLOBAL_SLICE[_spec.segment] = slice(_offset, _offset + _spec.size)
    _offset += _spec.size

NUM_TOKENS = _offset  # all entity tokens (excludes the model's learned global token)

# Token position -> Segment value; drives the type-embedding index buffer
TYPE_IDX: tuple[int, ...] = tuple(
    int(spec.segment) for spec in REGISTRY for _ in range(spec.size)
)


def segment_counts(mask: torch.Tensor) -> torch.Tensor:
    """Per-segment valid-token fractions from a token-tensor mask (B, NUM_TOKENS)
    -> (B, len(Segment)), registry order. Cardinality features for the global
    context (deck size, pile sizes, ...) that attention pooling blurs."""
    return torch.cat(
        [
            mask[:, GLOBAL_SLICE[spec.segment]].sum(dim=1, keepdim=True).float() / spec.size
            for spec in REGISTRY
        ],
        dim=1,
    )


# =============================================================================
# Action-pool views (selection heads pick from pools; every pool except MAP is
# a segment of the token tensor)
# =============================================================================

POOL_SEGMENT: dict[Pool, Segment] = {
    pool: Segment[pool.name] for pool in Pool if pool is not Pool.MAP
}

# Max entities per pool, indexed by Pool. MAP is not a token segment (it has its
# own per-column CNN), so its size comes from the map constant.
POOL_SIZE: list[int] = [
    SPEC[POOL_SEGMENT[pool]].size if pool is not Pool.MAP else MAP_WIDTH for pool in Pool
]
assert len(POOL_SIZE) == len(Pool), "POOL_SIZE must cover every Pool"

# Shop pools carry per-item prices, encoded alongside (not inside) the entity
# encodings; the value is the price tensor's field name on TGameState/TCoreOutput.
POOL_PRICE_FIELD: dict[Pool, str] = {
    Pool.SHOP_CARDS: "shop_card_prices",
    Pool.SHOP_RELICS: "shop_relic_prices",
    Pool.SHOP_POTIONS: "shop_potion_prices",
}
