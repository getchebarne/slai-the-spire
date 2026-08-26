from collections.abc import Iterator
from typing import get_args

import numpy as np
import torch
from slai import Combat
from slai import Effect
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import CostScope
from slai import GameState
from slai import members

from src.rl.constants import MAX_REWARD_CARD_BUNDLES
from src.rl.constants import MAX_REWARD_CARD_TOKENS
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.constants import MAX_SIZE_STASIS
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
    Slice(SliceKind.CARD_REWARD, MAX_REWARD_CARD_TOKENS),
    Slice(SliceKind.CARD_SHOP, MAX_SIZE_SHOP_CARDS),
    Slice(SliceKind.CARD_STASIS, MAX_SIZE_STASIS),
    Slice(SliceKind.CARD_EVENT, 1),  # event.roll_cards (We Meet Again stakes 1)
]
SLICE_KIND_CARDS = {slice_.kind for slice_ in SLICE_CARDS}
NUM_CARD_TOKENS = sum(slice_.size for slice_ in SLICE_CARDS)


# Enum maps, keyed by int(): the declaration-order discriminant, stable across
# engine content bumps (members() is alphabetical and shifts on every insertion)
_MAP_CARD_NAME = {card_name: int(card_name) for card_name in members(CardName)}
_MAP_CARD_KIND = {card_kind: int(card_kind) for card_kind in members(CardKind)}
_MAP_CARD_COLOR = {card_color: int(card_color) for card_color in members(CardColor)}
_MAP_CARD_RARITY = {card_rarity: int(card_rarity) for card_rarity in members(CardRarity)}
# CardCostKind is a union of flat variant classes (no discriminant): key on type
# (dir() on a UnionType yields nothing — building this from dir() silently
# produced a zero-width OHE.)
_MAP_CARD_COST_KIND = {cls: i for i, cls in enumerate(get_args(CardCostKind))}
_MAP_COST_SCOPE = {scope: int(scope) for scope in members(CostScope)}

# Cost
_COST_MIN = 0
_COST_MAX = 5
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1

# Per-card identity + energy. What the card DOES rides in its effect rows
# (effect.py), pooled into this token model-side
ENCODING_DIM_CARD = (
    len(_MAP_CARD_NAME)  # Name OHE
    + len(_MAP_CARD_KIND)  # Kind OHE
    + len(_MAP_CARD_COLOR)  # Color OHE
    + len(_MAP_CARD_RARITY)  # Rarity OHE
    + len(_MAP_CARD_COST_KIND)  # Cost kind OHE
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
    + len(_MAP_COST_SCOPE)  # Cost-override scope OHE (all-zero = no override)
    + MAX_REWARD_CARD_BUNDLES  # Reward bundle-index OHE — stamped OUTSIDE the cache
    + 1  # Staged for a multi-pick halt — stamped OUTSIDE the cache
)

# Bundle OHE lives at the row tail so the fill loop can stamp it after the cache
# copy: cached rows stay pure functions of (card, energy) — position bits never
# enter the cache (same pattern as map_.py's position stamp)
_OFF_PICKED = ENCODING_DIM_CARD - 1
_OFF_BUNDLE = _OFF_PICKED - MAX_REWARD_CARD_BUNDLES

# Balance assert: ties the dim sum to the write order of _encode_card_into, so a
# feature added to one but not the other fails loudly at import
_OFF_LEADING = (
    len(_MAP_CARD_NAME)
    + len(_MAP_CARD_KIND)
    + len(_MAP_CARD_COLOR)
    + len(_MAP_CARD_RARITY)
    + len(_MAP_CARD_COST_KIND)
    + _COST_VECTOR_DIM
)
_NUM_TAIL_SCALARS = 11  # the scalar block in _encode_card_into
assert ENCODING_DIM_CARD == _OFF_LEADING + (
    _NUM_TAIL_SCALARS + len(_MAP_COST_SCOPE) + MAX_REWARD_CARD_BUNDLES + 1
), "card row layout drifted: dim sum no longer matches encode_card_into"

# Per-card encoding cache, keyed by (card_identity, energy) — see card_identity
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000


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
    pos += 11

    # Cost-override scope OHE (Snecko Confusion / Snecko Oil / Madness / setup
    # picks); all-zero = no override. The overridden cost itself is card.cost
    if card.cost_override_scope is not None:
        out[pos + _MAP_COST_SCOPE[card.cost_override_scope]] = 1.0
    pos += len(_MAP_COST_SCOPE)

    # Bundle OHE tail: deliberately NOT written here (position-dependent)
    pos += MAX_REWARD_CARD_BUNDLES

    return pos


def card_identity(card: Card) -> tuple:
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


def card_selection_identity(card: Card) -> tuple:
    """Identity for selection dedup: the row identity AND the effects.

    Strictly finer than card_identity. Two same-name cards can carry different
    effects — the engine rewrites a single instance's DamagePhysical amount in
    place (utils.card_damage_delta, via Glass Knife decay and Ritual Dagger) —
    and those are NOT interchangeable picks, so the row key alone would mask one
    of the two plays off."""
    return (card_identity(card), tuple(card.effects))


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    """Write the full card encoding into `out`, cached by its read fields."""
    cache_key = (card_identity(card), energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        _encode_card_into(card, energy_current, 0, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_card_entities(
    kind: SliceKind, game_state: GameState, combat: Combat | None
) -> list[Card]:
    """The GameState card pile a card slice covers (used by the fill loop + masks dedup).

    `combat` is the caller's hoisted `game_state.combat`: every context access
    deep-clones the whole snapshot across the FFI, so it is fetched once per
    state, never per slice."""
    match kind:
        case SliceKind.CARD_HAND:
            return combat.hand if combat is not None else []
        case SliceKind.CARD_DRAW:
            # Head truncation, not the next-to-draw tail: CardMoveToHand.idx
            # indexes raw pile_draw positions, so slot i must equal engine index
            # i (overflow slots are warn-dropped by masks). Costs next-to-draw
            # visibility only when the pile exceeds the cap
            return combat.pile_draw[:MAX_SIZE_DRAW_PILE] if combat is not None else []
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
        case SliceKind.CARD_REWARD:
            reward = game_state.reward
            if reward is None:
                return []
            return [card for bundle in reward.cards for card in bundle]
        case SliceKind.CARD_SHOP:
            return game_state.shop.cards if game_state.shop is not None else []
        case SliceKind.CARD_EVENT:
            event = game_state.event
            if event is not None and not event.consumed:  # post-choice: the offer is dead
                return event.roll_cards
            return []
        case _:
            raise ValueError(f"Unsupported `SliceKind`: {kind}")


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
        # Deck hidden while fighting (visible via the piles); any other focus —
        # including a reward overlaying the ended combat — shows it. Equivalent
        # to derive_focus(game_state) is not Focus.COMBAT (reward is the only
        # context that outranks combat), from the already-hoisted snapshots
        reward = game_state.reward
        show_deck = _show_deck(game_state)
        # Cards staged in an open multi-pick halt. Empty on all but halt steps,
        # so the id set is built only when it can matter
        staged = game_state.effect_pending_picks
        staged_ids = {card.id for card in staged} if staged else None
        offset = 0
        for slice_ in SLICE_CARDS:
            # Skip the encode but advance offset so later slices align
            if slice_.kind is SliceKind.CARD_DECK and not show_deck:
                offset += slice_.size
                continue

            cards = clip_to_slice(_get_card_entities(slice_.kind, game_state, combat), slice_)

            for i, card in enumerate(cards):
                encode_card_into_w_cache(card, energy_current, np_out[b, offset + i])
                np_pad[b, offset + i] = True

            # Staged-pick flag, stamped after the cache copy (halt-dependent, so
            # it must never enter the row cache)
            if staged_ids:
                for i, card in enumerate(cards):
                    if card.id in staged_ids:
                        np_out[b, offset + i, _OFF_PICKED] = 1.0

            # Bundle-index OHE, stamped after the cache copy (position-dependent:
            # taking any card consumes its whole bundle, so membership matters)
            if slice_.kind is SliceKind.CARD_REWARD and reward is not None and cards:
                tok = 0
                for bundle_idx, bundle in enumerate(reward.cards):
                    ohe = min(bundle_idx, MAX_REWARD_CARD_BUNDLES - 1)
                    for _ in range(min(len(bundle), len(cards) - tok)):
                        np_out[b, offset + tok, _OFF_BUNDLE + ohe] = 1.0
                        tok += 1

            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )


def _show_deck(game_state: GameState) -> bool:
    """Deck hidden during combat (the piles carry it); a reward overlay outranks
    combat. Shared so the fill loop and the effect-owner walk cannot drift —
    a slot from one must always name a token the other masked True."""
    return game_state.reward is not None or game_state.combat is None


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, effects) for every card token that has effects.

    Same slice order, `_show_deck` rule and clip as encode_batch_cards, so a
    slot here always names a mask-True token."""
    combat = game_state.combat
    show_deck = _show_deck(game_state)
    offset = 0
    for slice_ in SLICE_CARDS:
        if slice_.kind is SliceKind.CARD_DECK and not show_deck:
            offset += slice_.size
            continue
        cards = clip_to_slice(_get_card_entities(slice_.kind, game_state, combat), slice_)
        for i, card in enumerate(cards):
            effects = card.effects  # bound once: each read clones the Vec
            if effects:
                yield offset + i, effects
        offset += slice_.size
