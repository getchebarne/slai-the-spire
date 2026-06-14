import warnings

import numpy as np
import torch
from slai import Card
from slai import GameState
from slai import CardColor
from slai import CardCostKind
from slai import CardKind
from slai import CardName
from slai import CardRarity
from slai import members

from src.rl.encoding.effect import ENCODING_DIM_EFFECTS
from src.rl.encoding.effect import encode_effects_into
from src.rl.index import CLASS_SEGMENTS
from src.rl.index import CLASS_SLICE
from src.rl.index import NUM_CLASS_TOKENS
from src.rl.index import EntityClass
from src.rl.index import Segment

# Enum maps
_MAP_CARD_NAME = {card_name: i for i, card_name in enumerate(members(CardName))}
_MAP_CARD_KIND = {card_kind: i for i, card_kind in enumerate(members(CardKind))}
_MAP_CARD_COLOR = {card_color: i for i, card_color in enumerate(members(CardColor))}
_MAP_CARD_RARITY = {card_rarity: i for i, card_rarity in enumerate(members(CardRarity))}
_MAP_CARD_COST_KIND = {
    card_cost_kind: i
    for i, card_cost_kind in enumerate(m for m in dir(CardCostKind) if not m.startswith("_"))
}

# Cost scalar
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

# Encoding cache: a card's encoding depends only on (card.identity_hash, energy_current)
_CARD_ROW_CACHE: dict[tuple[int, int], np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000

# Set tracking card segments that have triggered truncation warnings
_WARNED_TRUNCATED: set[Segment] = set()


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
    # Signed energy headroom: >0 surplus, <0 shortfall (abs() destroyed direction)
    out[pos + 8] = max(-_COST_MAX, min(energy_current - card.cost, _COST_MAX)) / _COST_MAX
    out[pos + 9] = max(_COST_MIN, min(card.cost_base, _COST_MAX)) / _COST_MAX
    out[pos + 10] = float(card.cost_zero_once)
    # Affordable bit (engine `playable` excludes energy; mirrors shop.py's flag)
    out[pos + 11] = float(card.cost <= energy_current)
    pos += 12

    return pos


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    cache_key = (card.identity_hash, energy_current)
    card_encoding = _CARD_ROW_CACHE.get(cache_key)
    if card_encoding is None:
        # Encode
        _encode_card_into(card, energy_current, 0, out)

        # Guard the cached master copy against potential writes
        out_copy = out.copy()
        out_copy.flags.writeable = False

        # Store master copy in the cache
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()

        _CARD_ROW_CACHE[cache_key] = out_copy
    else:
        out[:] = card_encoding


def encode_batch_cards(
    batch_game_state: list[GameState],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode every card segment (registry CARD class) into one concatenated
    (B, N_CARDS, ENCODING_DIM_CARD) tensor + mask; segments live at their
    index.CLASS_SLICE positions."""
    batch_size = len(batch_game_state)
    num_tokens = NUM_CLASS_TOKENS[EntityClass.CARD]

    # Allocate arrays
    x_out = np.zeros((batch_size, num_tokens, ENCODING_DIM_CARD), dtype=np.float32)
    x_pad = np.zeros((batch_size, num_tokens), dtype=bool)

    for b, game_state in enumerate(batch_game_state):
        for spec in CLASS_SEGMENTS[EntityClass.CARD]:
            cards = spec.getter(game_state)
            energy_current = spec.energy(game_state)
            if len(cards) > spec.size:
                # Truncate
                if spec.segment not in _WARNED_TRUNCATED:
                    _WARNED_TRUNCATED.add(spec.segment)
                    warnings.warn(
                        f"{spec.segment.name} pile of {len(cards)} cards truncated to"
                        f" encoder cap ({spec.size})"
                    )

                cards = cards[: spec.size]

            offset = CLASS_SLICE[spec.segment].start
            for i, card in enumerate(cards):
                encode_card_into_w_cache(card, energy_current, x_out[b, offset + i])

                # Tag mask
                x_pad[b, offset + i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
