import math
from enum import Enum
from typing import TypeAlias

import numpy as np
import torch

from src.game.const import MAX_SIZE_COMBAT_CARD_REWARD
from src.game.const import MAX_SIZE_DECK
from src.game.const import MAX_SIZE_DISC_PILE
from src.game.const import MAX_SIZE_DRAW_PILE
from src.game.const import MAX_SIZE_HAND
from src.game.core.effect import Effect
from src.game.core.effect import EffectSelectionType
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.factory.lib import FACTORY_LIB_CARD
from src.game.view.card import ViewCard


EffectKey: TypeAlias = tuple[EffectType, EffectTargetType, EffectSelectionType]
CardMetadata: TypeAlias = tuple[int, dict[EffectKey, int], dict[EffectKey, int]]


# TODO: put somewhere else?
class CardPile(Enum):
    HAND = "HAND"
    DRAW = "DRAW"
    DISC = "DISC"
    DECK = "DECK"
    COMBAT_REWARD = "COMBAT_REWARD"


_CARD_PILE_TO_MAX_SIZE = {
    CardPile.HAND: MAX_SIZE_HAND,
    CardPile.DRAW: MAX_SIZE_DRAW_PILE,
    CardPile.DISC: MAX_SIZE_DISC_PILE,
    CardPile.DECK: MAX_SIZE_DECK,
    CardPile.COMBAT_REWARD: MAX_SIZE_COMBAT_CARD_REWARD,
}


def _get_effect_key(effect: Effect) -> EffectKey:
    return (effect.type, effect.target_type, effect.selection_type)


def _get_card_metadata() -> CardMetadata:
    cost_max = -1
    effect_key_max = {}
    for _, card_factory in FACTORY_LIB_CARD.items():
        # Instantiate card
        card = card_factory(False)
        card_upgraded = card_factory(True)

        # Get maximum cost
        cost_max = max(cost_max, card.cost, card_upgraded.cost)

        # Get maximum value for each possible effect
        for effect in card.effects + card_upgraded.effects:
            effect_value = effect.value

            # If the effect's value is `None`, change it to 1 to signal its presence
            if effect_value is None:
                effect_value = 1

            effect_key = _get_effect_key(effect)
            if effect_key in effect_key_max:
                effect_key_max[effect_key] = max(effect_key_max[effect_key], effect_value)
            else:
                effect_key_max[effect_key] = effect_value

    # Also create a list to keep fixed positions for each effect key
    effect_key_pos = {effect_key: pos for pos, effect_key in enumerate(effect_key_max)}

    return cost_max, effect_key_max, effect_key_pos


def _compute_sqrt_bounds(
    key_to_max: dict[EffectKey, int], value_min: int = 0
) -> tuple[dict[EffectKey, tuple[int, int]], dict[EffectKey, int], int]:
    sqrt_min = int(math.sqrt(value_min))
    bounds = {}
    positions = {}

    pos_offset = 0
    for key, max_val in key_to_max.items():
        sqrt_max = int(math.sqrt(max_val))
        bounds[key] = (sqrt_min, sqrt_max)
        positions[key] = pos_offset
        pos_offset += sqrt_max - sqrt_min + 1

    return bounds, positions, pos_offset


# Get card metadata
_COST_MAX, _EFFECT_KEY_MAX, _EFFECT_KEY_POS = _get_card_metadata()

# Card names for one-hot encoding (base names without "+")
_CARD_NAMES = list(FACTORY_LIB_CARD.keys())
_NUM_CARD_NAMES = len(_CARD_NAMES)

# Pre-compute sqrt bounds for effect keys
_EFFECT_KEY_SQRT_BOUNDS, _EFFECT_KEY_SQRT_POS, _EFFECT_SQRT_TOTAL_DIM = _compute_sqrt_bounds(
    _EFFECT_KEY_MAX, value_min=0
)

# Pre-compute sqrt bounds for card cost
_COST_SQRT_MIN = int(math.sqrt(0))
_COST_SQRT_MAX = int(math.sqrt(_COST_MAX))
_COST_SQRT_DIM = _COST_SQRT_MAX - _COST_SQRT_MIN + 1


def _encode_view_card_into(out: np.ndarray, view_card: ViewCard, card_pile: CardPile) -> None:
    """Encode a card directly into a pre-allocated numpy array."""
    upgraded = view_card.name.endswith("+")
    pos = 0

    # Card name one-hot (strip "+" suffix for upgraded cards)
    base_name = view_card.name.rstrip("+")
    idx_name = _CARD_NAMES.index(base_name)
    out[idx_name] = 1.0
    pos += _NUM_CARD_NAMES

    # Cost sqrt one-hot
    cost_sqrt_value = int(math.sqrt(view_card.cost))
    cost_sqrt_value = max(min(cost_sqrt_value, _COST_SQRT_MAX), _COST_SQRT_MIN)

    # Position offsets (relative to after card name one-hot)
    pos_effects_sqrt = pos
    pos_effects_scalar = pos + _EFFECT_SQRT_TOTAL_DIM
    pos_cost_sqrt = pos_effects_scalar + len(_EFFECT_KEY_MAX)
    pos_scalars = pos_cost_sqrt + _COST_SQRT_DIM

    # Set cost sqrt one-hot
    out[pos_cost_sqrt + cost_sqrt_value - _COST_SQRT_MIN] = 1.0

    # Set scalar features
    out[pos_scalars] = view_card.cost / _COST_MAX
    out[pos_scalars + 1] = float(upgraded)
    out[pos_scalars + 2] = float(view_card.requires_target)
    out[pos_scalars + 3] = float(view_card.requires_discard)
    out[pos_scalars + 4] = float(view_card.exhaust)
    out[pos_scalars + 5] = float(view_card.innate)

    # Encode effects
    for effect in view_card.effects:
        effect_key = _get_effect_key(effect)

        effect_value = effect.value
        if effect_value is None:
            effect_value = 1.0

        # Sqrt one-hot encoding
        sqrt_min, sqrt_max = _EFFECT_KEY_SQRT_BOUNDS[effect_key]
        sqrt_value = int(math.sqrt(effect_value))
        sqrt_value = max(min(sqrt_value, sqrt_max), sqrt_min)
        sqrt_start_pos = pos_effects_sqrt + _EFFECT_KEY_SQRT_POS[effect_key]
        out[sqrt_start_pos + sqrt_value - sqrt_min] = 1.0

        # Scalar encoding
        scalar_pos = pos_effects_scalar + _EFFECT_KEY_POS[effect_key]
        out[scalar_pos] = effect_value / _EFFECT_KEY_MAX[effect_key]


def get_encoding_dim_card() -> int:
    # Calculate dimension from components
    return (
        _NUM_CARD_NAMES  # Card name one-hot
        + _EFFECT_SQRT_TOTAL_DIM  # Sqrt one-hot for effects
        + len(_EFFECT_KEY_MAX)  # Scalar for effects
        + _COST_SQRT_DIM  # Sqrt one-hot for cost
        + 6  # Scalars: cost, upgraded, requires_target, requires_discard, exhaust, innate
    )


_ENCODING_DIM_CARD = get_encoding_dim_card()


def encode_batch_view_cards(
    batch_view_cards: list[list[ViewCard]], card_pile: CardPile, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of card lists using NumPy pre-allocation."""
    max_size = _CARD_PILE_TO_MAX_SIZE[card_pile]
    batch_size = len(batch_view_cards)

    # Pre-allocate numpy arrays
    x_out = np.zeros((batch_size, max_size, _ENCODING_DIM_CARD), dtype=np.float32)
    x_mask_pad = np.zeros((batch_size, max_size), dtype=bool)

    for b, view_cards in enumerate(batch_view_cards):
        view_cards = view_cards[:max_size]
        for i, view_card in enumerate(view_cards):
            _encode_view_card_into(x_out[b, i], view_card, card_pile)
            x_mask_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_mask_pad).to(device),
    )
