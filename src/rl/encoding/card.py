import warnings
from collections.abc import Iterator
from enum import StrEnum
from enum import auto
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
from slai import EffectSetCostOverride
from slai import GameState
from slai import get_card_templates

from src.rl.constants import MAX_CARD_REWARD_ROLL
from src.rl.constants import MAX_COMBAT_CARD_REWARD
from src.rl.constants import MAX_PENDING_SELECTED
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCARD
from src.rl.constants import MAX_SIZE_CARD_PILE_DISCOVER
from src.rl.constants import MAX_SIZE_CARD_PILE_EVENT_ROLL
from src.rl.constants import MAX_SIZE_CARD_PILE_EXHAUST
from src.rl.constants import MAX_SIZE_CARD_PILE_HAND
from src.rl.constants import MAX_SIZE_CARD_PILE_NIGHTMARE
from src.rl.constants import MAX_SIZE_CARD_PILE_STASIS
from src.rl.constants import MAX_SIZE_CARD_PLAY_QUEUE
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.encoding.effect import catalog_rows
from src.rl.types import BatchEffectRows
from src.rl.types import BatchPriceRows
from src.rl.types import EffectRow
from src.rl.types import EffectSource
from src.rl.types import PriceRow
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator
from src.rl.utils import card_pile_draw_offset


SLICE_KINDS_CARD_REWARDS: tuple[SliceKind, ...] = (
    SliceKind.CARD_REWARD_0,
    SliceKind.CARD_REWARD_1,
    SliceKind.CARD_REWARD_2,
    SliceKind.CARD_REWARD_3,
    SliceKind.CARD_REWARD_4,
)

# Pack order within the card block: decision pools first, observational piles last
SLICE_KINDS_CARD = (
    SliceKind.CARD_PILE_HAND,
    SliceKind.CARD_PILE_DISCOVER,
    SliceKind.CARD_PENDING_SELECTED,
    *SLICE_KINDS_CARD_REWARDS,
    SliceKind.CARD_SHOP,
    SliceKind.CARD_PILE_EVENT_ROLL,
    SliceKind.CARD_MATCH_FLIPPED,
    SliceKind.CARD_PILE_STASIS,
    SliceKind.CARD_PLAY_QUEUE,
    SliceKind.CARD_PILE_NIGHTMARE,
    SliceKind.CARD_PILE_DRAW,
    SliceKind.CARD_PILE_DISCARD,
    SliceKind.CARD_PILE_EXHAUST,
    SliceKind.CARD_PILE_DECK,
)


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


# Per-card identity + energy
class CardChannel(StrEnum):
    NAME = auto()
    KIND = auto()
    COLOR = auto()
    RARITY = auto()
    COST_KIND = auto()
    COST_SCOPE = auto()  # all-zero = no override
    COST_VECTOR = auto()
    COST_SCALAR = auto()
    UPGRADED = auto()
    EXHAUST = auto()
    INNATE = auto()
    ETHEREAL = auto()
    RETAIN = auto()
    REQUIRES_TARGET = auto()
    PLAYABLE = auto()
    ENERGY_DELTA = auto()  # signed: energy_current - cost
    COST_BASE = auto()
    BOTTLED = auto()
    COST_OFFSET = auto()  # X-Cost offset
    FREE_TO_PLAY_ONCE = auto()  # Setup, Forethought: the 0 in `cost` lasts one play


CHANNEL_LAYOUT_CARD, _alloc = make_channel_allocator()
OFF_CARD_NAME = _alloc(CardChannel.NAME, len(_MAP_CARD_NAME))
OFF_CARD_KIND = _alloc(CardChannel.KIND, len(_MAP_CARD_KIND))
OFF_CARD_COLOR = _alloc(CardChannel.COLOR, len(_MAP_CARD_COLOR))
OFF_CARD_RARITY = _alloc(CardChannel.RARITY, len(_MAP_CARD_RARITY))
OFF_CARD_COST_KIND = _alloc(CardChannel.COST_KIND, len(_MAP_CARD_COST_KIND))
OFF_CARD_COST_SCOPE = _alloc(CardChannel.COST_SCOPE, len(_MAP_COST_SCOPE))
OFF_CARD_COST_VECTOR = _alloc(CardChannel.COST_VECTOR, _COST_VECTOR_DIM)
OFF_CARD_COST_SCALAR = _alloc(CardChannel.COST_SCALAR)
OFF_CARD_UPGRADED = _alloc(CardChannel.UPGRADED)
OFF_CARD_EXHAUST = _alloc(CardChannel.EXHAUST)
OFF_CARD_INNATE = _alloc(CardChannel.INNATE)
OFF_CARD_ETHEREAL = _alloc(CardChannel.ETHEREAL)
OFF_CARD_RETAIN = _alloc(CardChannel.RETAIN)
OFF_CARD_REQUIRES_TARGET = _alloc(CardChannel.REQUIRES_TARGET)
OFF_CARD_PLAYABLE = _alloc(CardChannel.PLAYABLE)
OFF_CARD_ENERGY_DELTA = _alloc(CardChannel.ENERGY_DELTA)
OFF_CARD_COST_BASE = _alloc(CardChannel.COST_BASE)
OFF_CARD_BOTTLED = _alloc(CardChannel.BOTTLED)
OFF_CARD_COST_OFFSET = _alloc(CardChannel.COST_OFFSET)
OFF_CARD_FREE_TO_PLAY_ONCE = _alloc(CardChannel.FREE_TO_PLAY_ONCE)

ENCODING_DIM_CARD = sum(w for _, w in CHANNEL_LAYOUT_CARD)

# Per-card encoding cache, keyed by (card_key, energy) — see card_key
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def _encode_card_into(card: Card, energy_current: int, out: np.ndarray) -> None:
    out[OFF_CARD_NAME + _MAP_CARD_NAME[card.name]] = 1.0
    out[OFF_CARD_KIND + _MAP_CARD_KIND[card.kind]] = 1.0
    out[OFF_CARD_COLOR + _MAP_CARD_COLOR[card.color]] = 1.0
    out[OFF_CARD_RARITY + _MAP_CARD_RARITY[card.rarity]] = 1.0
    out[OFF_CARD_COST_KIND + _MAP_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    if card.cost_override_scope is not None:
        out[OFF_CARD_COST_SCOPE + _MAP_COST_SCOPE[card.cost_override_scope]] = 1.0

    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    out[OFF_CARD_COST_VECTOR + cost] = 1.0

    out[OFF_CARD_COST_SCALAR] = cost / _COST_MAX
    out[OFF_CARD_UPGRADED] = float(card.upgraded)
    out[OFF_CARD_EXHAUST] = float(card.exhaust)
    out[OFF_CARD_INNATE] = float(card.innate)
    out[OFF_CARD_ETHEREAL] = float(card.ethereal)
    out[OFF_CARD_RETAIN] = float(card.retain)
    out[OFF_CARD_REQUIRES_TARGET] = float(card.requires_target)
    out[OFF_CARD_PLAYABLE] = float(card.playable)
    out[OFF_CARD_ENERGY_DELTA] = (
        max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    )
    out[OFF_CARD_COST_BASE] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[OFF_CARD_BOTTLED] = float(card.bottled)
    out[OFF_CARD_COST_OFFSET] = _cost_offset(card) / _COST_OFFSET_MAX
    out[OFF_CARD_FREE_TO_PLAY_ONCE] = float(card.free_to_play_once)


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
        card.free_to_play_once,
    )


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    # Try cache hit
    cache_key = (card_key(card), energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)

    # Cache miss
    if row is None:
        # Encode card
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        _encode_card_into(card, energy_current, row)

        # Write into cache
        row.flags.writeable = False 
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row

    # Write into output array
    out[:] = row


_REWARD_BUNDLE_OF_KIND = {kind: bundle for bundle, kind in enumerate(SLICE_KINDS_CARD_REWARDS)}

# The play queue and nightmare pile kinds that have already warned about overflow
_WARNED_OVERFLOW: set[SliceKind] = set()


def _capped(kind: SliceKind, cards: list[Card], cap: int) -> list[Card]:
    """`cards` cut to `cap`, warning once per kind when that drops any."""
    if len(cards) > cap and kind not in _WARNED_OVERFLOW:
        _WARNED_OVERFLOW.add(kind)
        warnings.warn(
            f"{kind.name} holds {len(cards)} cards; the ones past its cap of {cap} are dropped"
        )
    return cards[:cap]


def _get_card_entities(
    kind: SliceKind, game_state: GameState, combat: Combat | None
) -> list[Card]:
    """The GameState card pile a card kind names, truncated to the pile's cap
    (used by iter_packed_cards + masks dedup). A card past the cap has no token,
    so masks and prices cannot name it.

    `combat` is the caller's hoisted `game_state.combat`: every context access
    deep-clones the whole snapshot across the FFI, so it is fetched once per
    state, never per kind."""
    match kind:
        case SliceKind.CARD_PILE_HAND:
            return combat.card_pile_hand[:MAX_SIZE_CARD_PILE_HAND] if combat is not None else []
        case SliceKind.CARD_PILE_DRAW:
            if combat is None:
                return []
            # The top of the pile, still in engine order (bottom first)
            card_pile_draw = combat.card_pile_draw
            return card_pile_draw[card_pile_draw_offset(card_pile_draw) :]
        case SliceKind.CARD_PILE_DISCARD:
            return (
                combat.card_pile_discard[:MAX_SIZE_CARD_PILE_DISCARD] if combat is not None else []
            )
        case SliceKind.CARD_PILE_EXHAUST:
            return (
                combat.card_pile_exhaust[:MAX_SIZE_CARD_PILE_EXHAUST] if combat is not None else []
            )
        case SliceKind.CARD_PILE_STASIS:
            if combat is None:
                return []
            # Index-aligned with the monsters: a monster holding no card is None
            cards = [card for card in combat.card_pile_stasis if card is not None]
            return cards[:MAX_SIZE_CARD_PILE_STASIS]
        case SliceKind.CARD_PILE_DECK:
            return game_state.card_pile_deck[:MAX_SIZE_CARD_PILE_DECK]
        case SliceKind.CARD_PILE_DISCOVER:
            return (
                combat.card_pile_discover[:MAX_SIZE_CARD_PILE_DISCOVER]
                if combat is not None
                else []
            )
        case (
            SliceKind.CARD_REWARD_0
            | SliceKind.CARD_REWARD_1
            | SliceKind.CARD_REWARD_2
            | SliceKind.CARD_REWARD_3
            | SliceKind.CARD_REWARD_4
        ):
            reward = game_state.reward
            bundle = _REWARD_BUNDLE_OF_KIND[kind]
            if reward is None or bundle >= len(reward.cards):
                return []

            # Bundle 0 must hold The Library's single wide roll
            cap = MAX_CARD_REWARD_ROLL if bundle == 0 else MAX_COMBAT_CARD_REWARD
            return reward.cards[bundle][:cap]
        case SliceKind.CARD_SHOP:
            shop = game_state.shop
            return shop.cards[:MAX_SIZE_SHOP_CARDS] if shop is not None else []
        case SliceKind.CARD_PILE_EVENT_ROLL:
            event = game_state.event
            if event is not None and not event.consumed:
                return event.card_pile_event_roll[:MAX_SIZE_CARD_PILE_EVENT_ROLL]
            return []
        case SliceKind.CARD_MATCH_FLIPPED:
            event = game_state.event
            card = event.card_match_flipped if event is not None else None
            return [card] if card is not None else []
        case SliceKind.CARD_PLAY_QUEUE:
            if combat is None:
                return []
            return _capped(kind, combat.card_play_queue, MAX_SIZE_CARD_PLAY_QUEUE)
        case SliceKind.CARD_PILE_NIGHTMARE:
            if combat is None:
                return []
            return _capped(kind, combat.card_pile_nightmare, MAX_SIZE_CARD_PILE_NIGHTMARE)
        case SliceKind.CARD_PENDING_SELECTED:
            return game_state.effect_pending_selected[:MAX_PENDING_SELECTED]
        case _:
            raise ValueError(f"Unsupported `SliceKind`: {kind}")


def iter_packed_cards(
    game_state: GameState, combat: Combat | None
) -> Iterator[tuple[SliceKind, Card]]:
    for slice_kind in SLICE_KINDS_CARD:
        for card in _get_card_entities(slice_kind, game_state, combat):
            yield slice_kind, card


def encode_batch_cards(
    game_states: list[GameState], device: torch.device
) -> tuple[torch.Tensor, np.ndarray, BatchEffectRows, BatchPriceRows]:
    """(B, W, ENCODING_DIM_CARD) rows + the (B, W) `SliceKind` of each token + each state's
    (offset, source, effect) rows and its shop tokens' (offset, price) rows, keyed by the
    same w as the token they belong to; W is the batch's longest packed card sequence."""
    batch_tokens = []
    batch_energy_current = []
    batch_card_prices = []
    for game_state in game_states:
        combat = game_state.combat  # Hoisted; each read clones across the FFI
        batch_tokens.append(list(iter_packed_cards(game_state, combat)))
        batch_energy_current.append(combat.energy.energy_current if combat is not None else 0)
        shop = game_state.shop
        batch_card_prices.append(shop.card_prices if shop is not None else [])

    seqlen = max((len(tokens) for tokens in batch_tokens), default=0)
    np_out = np.zeros((len(game_states), seqlen, ENCODING_DIM_CARD), dtype=np.float32)
    np_idxs = np.full((len(game_states), seqlen), -1, dtype=np.int16)
    batch_effect_rows: BatchEffectRows = []
    batch_price_rows: BatchPriceRows = []

    for b, (tokens, energy_current) in enumerate(zip(batch_tokens, batch_energy_current)):
        effect_rows: list[EffectRow] = []
        price_rows: list[PriceRow] = []

        # Shop tokens come in shop order, so they take the parallel price list in step
        card_prices = iter(batch_card_prices[b])
        for w, (slice_kind, card) in enumerate(tokens):
            # Encode card
            encode_card_into_w_cache(card, energy_current, np_out[b, w])

            # Stamp `SliceKind`
            np_idxs[b, w] = slice_kind

            # Collect its price
            if slice_kind == SliceKind.CARD_SHOP:
                price_rows.append((w, next(card_prices)))

            # Collect Effects
            effects_play = card.effects_play  # Bound once; each read clones the Vec
            if effects_play:
                effect_rows += [(w, EffectSource.CARD_PLAY, effect) for effect in effects_play]
            effect_rows += [
                (w, EffectSource.CARD_DISCARD, effect) for effect in card.effects_discard
            ]
            effect_rows += [(w, EffectSource.CARD_DRAW, effect) for effect in card.effects_draw]
        batch_effect_rows.append(effect_rows)
        batch_price_rows.append(price_rows)

    return torch.from_numpy(np_out).to(device), np_idxs, batch_effect_rows, batch_price_rows
