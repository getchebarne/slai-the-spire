import warnings

import numpy as np
import torch
from slai import Card
from slai import CardColor
from slai import CardCostKind
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import members

from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into


# Enums
_MAP_CARD_NAME = {card_name: i for i, card_name in enumerate(members(CardName))}
_MAP_CARD_KIND = {card_kind: i for i, card_kind in enumerate(members(CardKind))}
_MAP_CARD_COLOR = {card_color: i for i, card_color in enumerate(members(CardColor))}
_MAP_CARD_RARITY = {card_rarity: i for i, card_rarity in enumerate(members(CardRarity))}
_MAP_CARD_COST_KIND = {card_cost_kind: i for i, card_cost_kind in enumerate(m for m in dir(CardCostKind) if not m.startswith("_"))}

# Cost
_COST_MIN = 0
_COST_MAX = 5
_COST_VECTOR_DIM = _COST_MAX - _COST_MIN + 1

# Encoding dimension
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
    + 1  # abs(energy_current - cost)
    + 1  # Cost base scalar
    + 1  # Cost zero-once (free-to-play-once)
    + ENCODING_DIM_EFFECTS  # Per-EffectKind feature blocks (effect.py)
)


def encode_card_into(card: Card, energy_current: int, pos: int, out: np.ndarray) -> int:
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

    # Scalars
    out[pos] = cost / _COST_MAX
    out[pos + 1] = float(card.upgraded)
    out[pos + 2] = float(card.exhaust)
    out[pos + 3] = float(card.innate)
    out[pos + 4] = float(card.ethereal)
    out[pos + 5] = float(card.retain)
    out[pos + 6] = float(card.requires_target)
    out[pos + 7] = float(card.playable)
    out[pos + 8] = min(abs(energy_current - card.cost), _COST_MAX) / _COST_MAX
    out[pos + 9] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[pos + 10] = float(card.cost_zero_once)
    pos += 11

    # Per-EffectKind effect blocks
    pos = encode_effects_into(card.effects, pos, out)

    return pos


def card_identity_key(card: Card) -> int:
    # Engine-computed fingerprint over every snapshot field encode_card_into reads
    # (incl. the exact effect records); one FFI getter instead of a ~16-field walk
    # + effects clone per call (~175k calls/rollout). Finer than the encoding
    # (unclamped cost), which can only miss a dedup, never alias two
    # encode-different cards. The energy-relative feature is caller state, not
    # card identity (excluded engine-side too).
    return card.identity_hash


# Per-identity row cache: a card's encoding depends only on (identity key, current
# energy), and piles are dominated by repeated identities — encode once, copy after.
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000  # safety valve; realistic size is a few thousand


def encode_card_row(card: Card, energy_current: int, out_row: np.ndarray) -> None:
    """Write a card's full encoding into `out_row` (a zeroed (ENCODING_DIM_CARD,)
    slice) through the identity cache."""
    key = (card_identity_key(card), energy_current)
    row = _CARD_ROW_CACHE.get(key)
    if row is None:
        encode_card_into(card, energy_current, 0, out_row)
        row = out_row.copy()
        row.flags.writeable = False  # guard the cached master copy
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[key] = row
    else:
        out_row[:] = row


# Caps we've already warned about truncating to — truncation must never be silent.
_WARNED_TRUNCATED: set[int] = set()


def encode_batch_cards(
    batch_cards: list[list[Card]],
    batch_energy_current: list[int],
    max_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (encodings, padding mask). Targeting is no longer derived here — the
    L3 target mask comes from the engine's legal actions (masks.py)."""
    batch_size = len(batch_cards)

    # Allocate arrays
    x_out = np.zeros((batch_size, max_size, ENCODING_DIM_CARD), dtype=np.float32)
    x_pad = np.zeros((batch_size, max_size), dtype=bool)

    for b, cards in enumerate(batch_cards):
        if len(cards) > max_size and max_size not in _WARNED_TRUNCATED:
            _WARNED_TRUNCATED.add(max_size)
            warnings.warn(f"Pile of {len(cards)} cards truncated to encoder cap {max_size}")
        cards = cards[:max_size]
        for i, card in enumerate(cards):
            encode_card_row(card, batch_energy_current[b], x_out[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
