import math
from enum import Enum
from typing import TypeAlias

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
from src.game.entity.card import CardColor  # TODO: should only import from `view`
from src.game.entity.card import CardRarity  # TODO: should only import from `view`
from src.game.entity.card import CardType  # TODO: should only import from `view`
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

# Pre-compute sqrt bounds for effect keys
_EFFECT_KEY_SQRT_BOUNDS, _EFFECT_KEY_SQRT_POS, _EFFECT_SQRT_TOTAL_DIM = _compute_sqrt_bounds(
    _EFFECT_KEY_MAX, value_min=0
)

# Pre-compute sqrt bounds for card cost
_COST_SQRT_MIN = int(math.sqrt(0))
_COST_SQRT_MAX = int(math.sqrt(_COST_MAX))
_COST_SQRT_DIM = _COST_SQRT_MAX - _COST_SQRT_MIN + 1


def _encode_view_card(view_card: ViewCard, card_pile: CardPile) -> list[float]:
    upgraded = view_card.name.endswith("+")  # TODO: add `upgraded` field

    # Cost sqrt one-hot
    cost_sqrt_value = int(math.sqrt(view_card.cost))
    cost_sqrt_value = max(min(cost_sqrt_value, _COST_SQRT_MAX), _COST_SQRT_MIN)
    cost_sqrt_one_hot = [0.0] * _COST_SQRT_DIM
    cost_sqrt_one_hot[cost_sqrt_value - _COST_SQRT_MIN] = 1.0

    encoding = (
        [0.0] * _EFFECT_SQRT_TOTAL_DIM  # Sqrt one-hot for effects
        + [0.0] * len(_EFFECT_KEY_MAX)  # Scalar for effects
        + cost_sqrt_one_hot  # Sqrt one-hot for cost
        + [
            view_card.cost / _COST_MAX,  # Scalar for cost
            float(upgraded),
            float(view_card.requires_target),
            float(view_card.requires_discard),
            float(view_card.exhaust),
            float(view_card.innate),
        ]
    )

    for effect in view_card.effects:
        effect_key = _get_effect_key(effect)

        # Get effect value (None -> 1 to signal presence)
        effect_value = effect.value
        if effect_value is None:
            effect_value = 1.0

        # Sqrt one-hot encoding
        sqrt_min, sqrt_max = _EFFECT_KEY_SQRT_BOUNDS[effect_key]
        sqrt_value = int(math.sqrt(effect_value))
        sqrt_value = max(min(sqrt_value, sqrt_max), sqrt_min)
        sqrt_start_pos = _EFFECT_KEY_SQRT_POS[effect_key]
        sqrt_offset = sqrt_value - sqrt_min
        encoding[sqrt_start_pos + sqrt_offset] = 1.0

        # Scalar encoding (after sqrt one-hots)
        scalar_pos = _EFFECT_SQRT_TOTAL_DIM + _EFFECT_KEY_POS[effect_key]
        encoding[scalar_pos] = effect_value / _EFFECT_KEY_MAX[effect_key]

    return encoding


def get_encoding_dim_card() -> int:
    view_card_dummy = ViewCard(
        "Dummy",
        CardColor.GREEN,
        CardType.ATTACK,
        CardRarity.BASIC,
        1,
        [],
        False,
        False,
        False,
        False,
        False,
    )

    encoding_card_dummy = _encode_view_card(view_card_dummy, CardPile.HAND)
    return len(encoding_card_dummy)


# TODO: handle these constants better
_ENCODING_CARD_PAD = [0.0] * get_encoding_dim_card()


def encode_batch_view_cards(
    batch_view_cards: list[list[ViewCard]], card_pile: CardPile, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    # Get pile max size
    max_size = _CARD_PILE_TO_MAX_SIZE[card_pile]

    # Initialize empty lists to store encodings for each batch
    x_out = []
    x_mask_pad = []

    # Iterate over batches
    for view_cards in batch_view_cards:
        # Truncate to max size TODO: add log
        view_cards = view_cards[:max_size]

        encoding_cards = []
        for view_card in view_cards:
            encoding_card = _encode_view_card(view_card, card_pile)
            encoding_cards.append(encoding_card)

        # Pad to `max_size`
        num_pad = max_size - len(view_cards)
        encoding_cards += [_ENCODING_CARD_PAD] * num_pad

        # Append batch encodings and padding mask
        x_out.append(encoding_cards)
        x_mask_pad.append([True] * len(view_cards) + [False] * num_pad)

    return (
        torch.tensor(x_out, dtype=torch.float32, device=device),
        torch.tensor(x_mask_pad, dtype=torch.bool, device=device),
    )
