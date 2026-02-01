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


# Get card metadata
_COST_MAX, _EFFECT_KEY_MAX, _EFFECT_KEY_POS = _get_card_metadata()


def _encode_view_card(view_card: ViewCard, card_pile: CardPile) -> list[float]:
    upgraded = view_card.name.endswith("+")  # TODO: add `upgraded` field
    encoding = [0] * len(_EFFECT_KEY_MAX) + [
        view_card.cost / _COST_MAX,
        float(upgraded),
        float(view_card.requires_target),
        float(view_card.requires_discard),
        float(view_card.exhaust),
        float(view_card.innate),
        float(card_pile == CardPile.HAND),
        float(card_pile == CardPile.DRAW),
        float(card_pile == CardPile.DISC),
        float(card_pile == CardPile.DECK),
        float(card_pile == CardPile.COMBAT_REWARD),
    ]

    for effect in view_card.effects:
        effect_key = _get_effect_key(effect)
        effect_key_pos = _EFFECT_KEY_POS[effect_key]

        # Get effect value
        effect_value = effect.value
        if effect_value is None:
            effect_value = 1.0

        encoding[effect_key_pos] = effect_value / _EFFECT_KEY_MAX[effect_key]

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
