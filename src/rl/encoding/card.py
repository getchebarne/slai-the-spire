
import numpy as np
from slai import Card
from slai import CardCostKindXCost

from src.rl.schema.card import COST_MAX
from src.rl.schema.card import COST_MIN
from src.rl.schema.card import COST_OFFSET_MAX
from src.rl.schema.card import ENCODING_DIM_CARD
from src.rl.schema.card import MAP_CARD_COLOR
from src.rl.schema.card import MAP_CARD_COST_KIND
from src.rl.schema.card import MAP_CARD_KIND
from src.rl.schema.card import MAP_CARD_NAME
from src.rl.schema.card import MAP_CARD_RARITY
from src.rl.schema.card import MAP_COST_SCOPE
from src.rl.schema.card import MAP_PLAY_SOURCE
from src.rl.schema.card import OFF_CARD_BOTTLED
from src.rl.schema.card import OFF_CARD_COLOR
from src.rl.schema.card import OFF_CARD_COST_BASE
from src.rl.schema.card import OFF_CARD_COST_KIND
from src.rl.schema.card import OFF_CARD_COST_OFFSET
from src.rl.schema.card import OFF_CARD_COST_SCALAR
from src.rl.schema.card import OFF_CARD_COST_SCOPE
from src.rl.schema.card import OFF_CARD_COST_VECTOR
from src.rl.schema.card import OFF_CARD_ENERGY_DELTA
from src.rl.schema.card import OFF_CARD_ETHEREAL
from src.rl.schema.card import OFF_CARD_EXHAUST
from src.rl.schema.card import OFF_CARD_FREE_TO_PLAY_ONCE
from src.rl.schema.card import OFF_CARD_INNATE
from src.rl.schema.card import OFF_CARD_KIND
from src.rl.schema.card import OFF_CARD_NAME
from src.rl.schema.card import OFF_CARD_PLAY_SOURCE
from src.rl.schema.card import OFF_CARD_PLAYABLE
from src.rl.schema.card import OFF_CARD_RARITY
from src.rl.schema.card import OFF_CARD_REQUIRES_TARGET
from src.rl.schema.card import OFF_CARD_RETAIN
from src.rl.schema.card import OFF_CARD_UPGRADED


# Per-card encoding cache, keyed by (card_key, energy) — see card_key
_CARD_ROW_CACHE: dict[tuple, np.ndarray] = {}
_CARD_ROW_CACHE_MAX = 100_000


def _cost_offset(card: Card) -> float:
    return float(card.cost_kind.offset if isinstance(card.cost_kind, CardCostKindXCost) else 0)


def _encode_card_into(card: Card, energy_current: int, out: np.ndarray) -> None:
    out[OFF_CARD_NAME + MAP_CARD_NAME[card.name]] = 1.0
    out[OFF_CARD_KIND + MAP_CARD_KIND[card.kind]] = 1.0
    out[OFF_CARD_COLOR + MAP_CARD_COLOR[card.color]] = 1.0
    out[OFF_CARD_RARITY + MAP_CARD_RARITY[card.rarity]] = 1.0
    out[OFF_CARD_COST_KIND + MAP_CARD_COST_KIND[type(card.cost_kind)]] = 1.0
    if card.cost_override_scope is not None:
        out[OFF_CARD_COST_SCOPE + MAP_COST_SCOPE[card.cost_override_scope]] = 1.0

    cost = max(COST_MIN, min(card.cost, COST_MAX))
    out[OFF_CARD_COST_VECTOR + cost] = 1.0

    out[OFF_CARD_COST_SCALAR] = cost / COST_MAX
    out[OFF_CARD_UPGRADED] = float(card.upgraded)
    out[OFF_CARD_EXHAUST] = float(card.exhaust)
    out[OFF_CARD_INNATE] = float(card.innate)
    out[OFF_CARD_ETHEREAL] = float(card.ethereal)
    out[OFF_CARD_RETAIN] = float(card.retain)
    out[OFF_CARD_REQUIRES_TARGET] = float(card.requires_target)
    out[OFF_CARD_PLAYABLE] = float(card.playable)
    out[OFF_CARD_ENERGY_DELTA] = (
        max(-COST_MAX, min(energy_current - card.cost, COST_MAX)) / COST_MAX
    )
    out[OFF_CARD_COST_BASE] = max(COST_MIN, min(card.cost_base, COST_MAX)) / COST_MAX
    out[OFF_CARD_BOTTLED] = float(card.bottled)
    out[OFF_CARD_COST_OFFSET] = _cost_offset(card) / COST_OFFSET_MAX
    out[OFF_CARD_FREE_TO_PLAY_ONCE] = float(card.free_to_play_once)
    if card.play_source is not None:
        out[OFF_CARD_PLAY_SOURCE + MAP_PLAY_SOURCE[card.play_source]] = 1.0


def card_key(card: Card) -> tuple:
    """Return the fields _encode_card_into reads, minus energy.

    Two Cards with equal identity encode the same ROW at any given energy. It keys the row cache
    and, with effect_key, masks' identity dedup; Effects are rows of their own, so absent.

    Explicit rather than hash(Card): live Entities carry a unique arena `id` that participates in
    the engine's __hash__, so hashing the object would make every Card distinct and silently kill
    the cache.

    Parameters
    ----------
    card : Card
        The Card to key.

    Returns
    -------
    tuple
        The Card's identity key: those fields' values.
    """

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
        card.play_source,
    )


def encode_card_into_w_cache(card: Card, energy_current: int, out: np.ndarray) -> None:
    # Try cache hit
    cache_key = (card_key(card), energy_current)
    row = _CARD_ROW_CACHE.get(cache_key)

    # Cache miss
    if row is None:
        # Encode Card
        row = np.zeros(ENCODING_DIM_CARD, dtype=np.float32)
        _encode_card_into(card, energy_current, row)

        # Write into cache
        row.flags.writeable = False
        if len(_CARD_ROW_CACHE) >= _CARD_ROW_CACHE_MAX:
            _CARD_ROW_CACHE.clear()
        _CARD_ROW_CACHE[cache_key] = row

    # Write into output array
    out[:] = row
