import warnings

import numpy as np
import torch
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import GameState
from slai import Screen
from slai import members

from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DISCOVER
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_EXHAUST
from src.rl.constants import MAX_SIZE_HAND
from src.rl.constants import MAX_SIZE_REWARD_CARDS
from src.rl.constants import MAX_SIZE_SHOP_CARDS
from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.types import Slice
from src.rl.types import SliceKind


# Order = fill order = Core's global-offset order
SLICE_CARDS = [
    Slice(SliceKind.CARD_HAND, MAX_SIZE_HAND),
    Slice(SliceKind.CARD_DRAW, MAX_SIZE_DRAW_PILE),
    Slice(SliceKind.CARD_DISCARD, MAX_SIZE_DISC_PILE),
    Slice(SliceKind.CARD_EXHAUST, MAX_SIZE_EXHAUST),
    Slice(SliceKind.CARD_DECK, MAX_SIZE_DECK),
    Slice(SliceKind.CARD_DISCOVER, MAX_SIZE_DISCOVER),
    Slice(SliceKind.CARD_REWARD, MAX_SIZE_REWARD_CARDS),
    Slice(SliceKind.CARD_SHOP, MAX_SIZE_SHOP_CARDS),
]
SLICE_KIND_CARDS = {slice_.kind for slice_ in SLICE_CARDS}
NUM_CARD_TOKENS = sum(slice_.size for slice_ in SLICE_CARDS)


# Enum maps
_MAP_CARD_NAME = {card_name: i for i, card_name in enumerate(members(CardName))}
_MAP_CARD_KIND = {card_kind: i for i, card_kind in enumerate(members(CardKind))}
_MAP_CARD_COLOR = {card_color: i for i, card_color in enumerate(members(CardColor))}
_MAP_CARD_RARITY = {card_rarity: i for i, card_rarity in enumerate(members(CardRarity))}
_MAP_CARD_COST_KIND = {
    card_cost_kind: i
    for i, card_cost_kind in enumerate(m for m in dir(CardCostKind) if not m.startswith("_"))
}

# Cost
_COST_MIN = 0
_COST_MAX = 5
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1

# Per-card identity + energy + per-EffectKind effect features (effects carry the live adjustment).
ENCODING_DIM_CARD = (
    len(_MAP_CARD_NAME)  # Name OHE
    + len(_MAP_CARD_KIND)  # Kind OHE
    + len(_MAP_CARD_COLOR)  # Color OHE
    + len(_MAP_CARD_RARITY)  # Rarity OHE
    + len(_MAP_CARD_COST_KIND)  # Cost kind OHE
    + _COST_VECTOR_DIM  # Cost vector
    + ENCODING_DIM_EFFECTS  # Per-EffectKind feature blocks (effect.py)
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
    + 1  # Cost zero-once (free-to-play-once)
    + 1  # Affordable (cost <= energy_current)
)

# Per-card encoding cache keyed by (identity_hash, energy); identity_hash now varies with modifiers.
_CARD_ROW_CACHE: dict[tuple[int, int], np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000

# Set tracking card slices that have triggered truncation warnings
_WARNED_TRUNCATED: set[SliceKind] = set()

# Screens where the full owned deck is shown; hidden in combat (visible via the piles).
_DECK_SCREENS = frozenset(
    {Screen.Map, Screen.Chest, Screen.RestSite, Screen.Shop, Screen.Reward, Screen.Event}
)


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
    out[pos + _MAP_CARD_COST_KIND[type(card.cost_kind).__name__]] = 1.0
    pos += len(_MAP_CARD_COST_KIND)

    # Cost vector
    cost = max(_COST_MIN, min(card.cost, _COST_MAX))
    out[pos + cost] = 1.0
    pos += _COST_VECTOR_DIM

    # Per-EffectKind effect blocks
    pos = encode_effects_into(card.effects, pos, out)

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
    out[pos + 10] = float(card.cost_zero_once)
    out[pos + 11] = float(card.cost <= energy_current)
    pos += 12

    return pos


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    """Write the full card encoding into `out`, cached by (identity_hash, energy)."""
    cache_key = (card.identity_hash, energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)
    if row is None:
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        _encode_card_into(card, energy_current, 0, row)
        row.flags.writeable = False  # guard the cached master copy
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row
    out[:] = row


def _get_card_entities(kind: SliceKind, game_state: GameState) -> list:
    """The GameState card pile a card slice covers (used by the fill loop + masks dedup)."""
    match kind:
        case SliceKind.CARD_HAND:
            return game_state.hand
        case SliceKind.CARD_DRAW:
            return game_state.pile_draw[-MAX_SIZE_DRAW_PILE:]  # next-to-draw tail
        case SliceKind.CARD_DISCARD:
            return game_state.pile_discard
        case SliceKind.CARD_EXHAUST:
            return game_state.pile_exhaust
        case SliceKind.CARD_DECK:
            return game_state.deck  # master deck
        case SliceKind.CARD_DISCOVER:
            return game_state.discover
        case SliceKind.CARD_REWARD:
            return game_state.reward.cards if game_state.reward is not None else []
        case SliceKind.CARD_SHOP:
            return game_state.shop.cards if game_state.shop is not None else []
        case _:
            raise ValueError(f"not a card slice: {kind}")


def encode_batch_cards(
    batch_game_state: list[GameState],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode all card slices into one (B, NUM_CARD_TOKENS, ENCODING_DIM_CARD) tensor + mask."""
    batch_size = len(batch_game_state)
    np_out = np.zeros((batch_size, NUM_CARD_TOKENS, ENCODING_DIM_CARD), dtype=np.float32)
    np_pad = np.zeros((batch_size, NUM_CARD_TOKENS), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        energy_current = game_state.energy.energy_current
        offset = 0
        for slice_ in SLICE_CARDS:
            # Deck hidden in combat; skip the encode but advance offset so later slices align
            if slice_.kind is SliceKind.CARD_DECK and game_state.screen not in _DECK_SCREENS:
                offset += slice_.size
                continue

            cards = _get_card_entities(slice_.kind, game_state)
            if len(cards) > slice_.size:
                if slice_.kind not in _WARNED_TRUNCATED:
                    _WARNED_TRUNCATED.add(slice_.kind)
                    warnings.warn(
                        f"{slice_.kind.name} pile of {len(cards)} cards truncated to"
                        f" encoder cap ({slice_.size})"
                    )
                cards = cards[: slice_.size]

            for i, card in enumerate(cards):
                encode_card_into_w_cache(card, energy_current, np_out[b, offset + i])
                np_pad[b, offset + i] = True
            offset += slice_.size

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
    )
