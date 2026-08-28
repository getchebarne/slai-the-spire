from collections.abc import Iterator
from typing import get_args

import numpy as np
import torch
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardCostKindXCost
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import Combat
from slai import CostScope
from slai import Effect
from slai import EffectSetCostOverride
from slai import GameState
from slai import get_card_templates

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import MAX_SIZE_STASIS
from src.rl.constants import REWARD_REGION_WIDTHS
from src.rl.encoding.effect import catalog_rows
from src.rl.types import REWARD_SLICE_KINDS
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import clip_to_slice


# Order = fill order = Core's global-offset order
SLICE_CARDS = [
    Slice(SliceKind.CARD_HAND, MAX_SIZE_HAND),
    Slice(SliceKind.CARD_DRAW, MAX_SIZE_DRAW_PILE),
    Slice(SliceKind.CARD_DISCARD, MAX_SIZE_DISC_PILE),
    Slice(SliceKind.CARD_EXHAUST, MAX_SIZE_EXHAUST),
    Slice(SliceKind.CARD_DECK, MAX_SIZE_DECK),
    Slice(SliceKind.CARD_DISCOVER, MAX_SIZE_DISCOVER),
    *(
        Slice(kind, width)
        for kind, width in zip(REWARD_SLICE_KINDS, REWARD_REGION_WIDTHS, strict=True)
    ),
    Slice(SliceKind.CARD_SHOP, MAX_SIZE_SHOP_CARDS),
    Slice(SliceKind.CARD_STASIS, MAX_SIZE_STASIS),
    Slice(SliceKind.CARD_EVENT, 1),
    Slice(SliceKind.CARD_PENDING_SELECTED, MAX_SIZE_HAND),
]
SLICE_KIND_CARDS = {slice_.kind for slice_ in SLICE_CARDS}
NUM_CARD_TOKENS = sum(slice_.size for slice_ in SLICE_CARDS)


# Enum maps, keyed by int(): the declaration-order discriminant
_MAP_CARD_NAME = {card_name: int(card_name) for card_name in CardName.members()}
_MAP_CARD_KIND = {card_kind: int(card_kind) for card_kind in CardKind.members()}
_MAP_CARD_COLOR = {card_color: int(card_color) for card_color in CardColor.members()}
_MAP_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in CardRarity.members()}
_MAP_COST_SCOPE = {scope: int(scope) for scope in CostScope.members()}
_MAP_CARD_COST_KIND = {cls: i for i, cls in enumerate(get_args(CardCostKind))}

# Cost, derived: the catalog's widest base cost or override amount
_COST_MIN = 0
_COST_MAX = max(
    max(card_template.cost_base for card_template in get_card_templates()),
    max(
        effecet.amount
        for _, effecet in catalog_rows()
        if isinstance(effecet, EffectSetCostOverride)
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

# Per-card identity + energy. What the card DOES rides in its effect rows
# (effect.py), pooled into this token model-side
ENCODING_DIM_CARD = (
    len(_MAP_CARD_NAME)  # Name OHE
    + len(_MAP_CARD_KIND)  # Kind OHE
    + len(_MAP_CARD_COLOR)  # Color OHE
    + len(_MAP_CARD_RARITY)  # Rarity OHE
    + len(_MAP_CARD_COST_KIND)  # Cost kind OHE
    + len(_MAP_COST_SCOPE)  # Cost-override scope OHE (all-zero = no override)
    + _COST_VECTOR_DIM  # Cost vector
    + 1  # Cost scalar
    + 1  # Upgraded
    + 1  # Exhaust
    + 1  # Innate
    + 1  # Ethereal
    + 1  # Retain
    + 1  # Requires target
    + 1  # Playable
    + 1  # Signed energy delta (energy_current - cost)
    + 1  # Cost base scalar
    + 1  # Bottled
    + 1  # X-Cost offset
)

# Per-card encoding cache, keyed by (card_key, energy) — see card_key
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def _encode_card_into(card: Card, energy_current: int, pos: int, out: np.ndarray) -> int:
    # Name OHE
    out[pos + _MAP_CARD_NAME[card.name]] = 1.0
    pos += len(_MAP_CARD_NAME)

    # Kind OHE
    out[pos + _MAP_CARD_KIND[card.kind]] = 1.0
    pos += len(_MAP_CARD_KIND)

    # Color OHE
    out[pos + _MAP_CARD_COLOR[card.color]] = 1.0
    pos += len(_MAP_CARD_COLOR)

    # Rarity OHE
    out[pos + _MAP_CARD_RARITY[card.rarity]] = 1.0
    pos += len(_MAP_CARD_RARITY)

    # Cost kind OHE
    out[pos + _MAP_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    pos += len(_MAP_CARD_COST_KIND)

    # Cost-override scope OHE
    if card.cost_override_scope is not None:
        out[pos + _MAP_COST_SCOPE[card.cost_override_scope]] = 1.0
    pos += len(_MAP_COST_SCOPE)

    # Cost vector
    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    out[pos + cost] = 1.0
    pos += _COST_VECTOR_DIM

    # Scalars
    out[pos] = cost / _COST_MAX
    out[pos + 1] = float(card.upgraded)
    out[pos + 2] = float(card.exhaust)
    out[pos + 3] = float(card.innate)
    out[pos + 4] = float(card.ethereal)
    out[pos + 5] = float(card.retain)
    out[pos + 6] = float(card.requires_target)
    out[pos + 7] = float(card.playable)
    out[pos + 8] = max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    out[pos + 9] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[pos + 10] = float(card.bottled)
    out[pos + 11] = _cost_offset(card) / _COST_OFFSET_MAX
    pos += 12

    return pos


def card_key(card: Card) -> tuple:
    """The fields _encode_card_into reads, minus energy — two cards with equal
    identity encode the same ROW at any given energy. This is the row-cache key
    and nothing else; effects are absent because they left the row.

    Explicit rather than hash(card): live entities carry a unique arena `id`
    that participates in the engine's __hash__, so hashing the object would make
    every card distinct and silently kill the cache."""

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
    )


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    cache_key = (card_key(card), energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        pos = _encode_card_into(card, energy_current, 0, row)
        assert pos == ENCODING_DIM_CARD, "card row layout drifted from ENCODING_DIM_CARD"
        row.flags.writeable = False  # guard the cached master copy
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row
    out[:] = row


_REWARD_BUNDLE_OF_KIND = {kind: bundle for bundle, kind in enumerate(REWARD_SLICE_KINDS)}


def _get_card_entities(
    kind: SliceKind, game_state: GameState, combat: Combat | None
) -> list[Card]:
    """The GameState card pile a card slice covers (used by slice_slots + masks dedup).

    `combat` is the caller's hoisted `game_state.combat`: every context access
    deep-clones the whole snapshot across the FFI, so it is fetched once per
    state, never per slice."""
    match kind:
        case SliceKind.CARD_HAND:
            return combat.hand if combat is not None else []
        case SliceKind.CARD_DRAW:
            return combat.pile_draw if combat is not None else []
        case SliceKind.CARD_DISCARD:
            return combat.pile_discard if combat is not None else []
        case SliceKind.CARD_EXHAUST:
            return combat.pile_exhaust if combat is not None else []
        case SliceKind.CARD_STASIS:
            return combat.pile_stasis if combat is not None else []
        case SliceKind.CARD_DECK:
            return game_state.deck
        case SliceKind.CARD_DISCOVER:
            return combat.discover if combat is not None else []
        case (
            SliceKind.CARD_REWARD_0
            | SliceKind.CARD_REWARD_1
            | SliceKind.CARD_REWARD_2
            | SliceKind.CARD_REWARD_3
            | SliceKind.CARD_REWARD_4
        ):
            reward = game_state.reward
            if reward is None:
                return []
            bundle = _REWARD_BUNDLE_OF_KIND[kind]
            return reward.cards[bundle] if bundle < len(reward.cards) else []
        case SliceKind.CARD_SHOP:
            return game_state.shop.cards if game_state.shop is not None else []
        case SliceKind.CARD_EVENT:
            event = game_state.event
            if event is not None and not event.consumed:
                return event.roll_cards
            return []
        case SliceKind.CARD_PENDING_SELECTED:
            return game_state.effect_pending_selected
        case _:
            raise ValueError(f"Unsupported `SliceKind`: {kind}")


def slice_slots(
    slice_: Slice, game_state: GameState, combat: Combat | None
) -> Iterator[tuple[int, Card]]:
    """(slot-within-slice, Card) for one slice — the single placement authority."""
    yield from enumerate(
        clip_to_slice(_get_card_entities(slice_.kind, game_state, combat), slice_)
    )


def encode_batch_cards(
    batch_game_state: list[GameState],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all card slices into one (B, NUM_CARD_TOKENS, ENCODING_DIM_CARD) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_CARD_TOKENS, ENCODING_DIM_CARD), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_CARD_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        combat = game_state.combat
        energy_current = combat.energy.energy_current if combat is not None else 0
        offset = 0
        for slice_ in SLICE_CARDS:
            for slot, card in slice_slots(slice_, game_state, combat):
                encode_card_into_w_cache(card, energy_current, np_out[b, offset + slot])
                np_pad[b, offset + slot] = True

            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, effects) for every card token that has effects.

    Same slice order and clip as encode_batch_cards, so a slot here always
    names a mask-True token."""
    combat = game_state.combat
    offset = 0
    for slice_ in SLICE_CARDS:
        for slot, card in slice_slots(slice_, game_state, combat):
            effects = card.effects  # bound once: each read clones the Vec
            if effects:
                yield offset + slot, effects
        offset += slice_.size
