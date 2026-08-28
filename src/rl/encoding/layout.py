"""The layout spine: entity-type order, token universe, owner-segment spaces.

Channels, Slice tables, and weights stay in their modules; this file owns only
the agreements between them: which entity types exist, in what order, which
token slices each contributes, and how the flat owner-segment spaces index
their tokens. Anything that cannot derive is pinned by an import-time assert.
"""

from enum import IntEnum

from src.rl.encoding.card import SLICE_CARDS
from src.rl.encoding.character import SLICE_CHARACTER
from src.rl.encoding.event import SLICE_EVENTS
from src.rl.encoding.map_ import SLICE_ROOMS
from src.rl.encoding.monster import SLICE_MONSTERS
from src.rl.encoding.potion import SLICE_POTIONS
from src.rl.encoding.relic import SLICE_RELICS
from src.rl.types import Slice
from src.rl.types import TGameState
from src.rl.types import TPadded


class EntityIdx(IntEnum):
    """The one entity-type order: tensorclass fields, projector layouts, and
    Core's token cat all follow it."""

    CARD = 0
    RELIC = 1
    POTION = 2
    MONSTER = 3
    EVENT_OPTION = 4
    CHARACTER = 5


# Token slices per entity type, fill order
ENTITIES: dict[EntityIdx, tuple[Slice, ...]] = {
    EntityIdx.CARD: tuple(SLICE_CARDS),
    EntityIdx.RELIC: tuple(SLICE_RELICS),
    EntityIdx.POTION: tuple(SLICE_POTIONS),
    EntityIdx.MONSTER: tuple(SLICE_MONSTERS),
    EntityIdx.EVENT_OPTION: tuple(SLICE_EVENTS),
    EntityIdx.CHARACTER: tuple(SLICE_CHARACTER),
}

# Global token universe in cat order; Rooms last (GNN node features, no row schema)
SLICE_ALL: tuple[Slice, ...] = tuple(
    slice_ for slices in ENTITIES.values() for slice_ in slices
) + tuple(SLICE_ROOMS)


def get_encodings_for_idx(t: TGameState, idx: EntityIdx) -> TPadded:
    """The entity type's encoder (x, mask) tokens."""
    match idx:
        case EntityIdx.CARD:
            return t.cards
        case EntityIdx.RELIC:
            return t.relics
        case EntityIdx.POTION:
            return t.potions
        case EntityIdx.MONSTER:
            return t.monsters
        case EntityIdx.EVENT_OPTION:
            return t.event_options
        case EntityIdx.CHARACTER:
            return t.character
        case _:
            raise ValueError(f"no entity tokens for {idx}")


# Global token offset per entity type (ROOMS follow last, so entity tokens prefix the space)
_WIDTHS = [sum(slice_.size for slice_ in slices) for slices in ENTITIES.values()]
TOKEN_OFFSET: dict[EntityIdx, int] = {idx: sum(_WIDTHS[:i]) for i, idx in enumerate(ENTITIES)}
NUM_ENTITY_TOKENS = sum(_WIDTHS)

# The pending halt owns no token: one virtual slot past every entity token
EFFECT_SEGMENT_PENDING = NUM_ENTITY_TOKENS


assert len({slice_.kind for slice_ in SLICE_ALL}) == len(SLICE_ALL), "duplicate SliceKind"
