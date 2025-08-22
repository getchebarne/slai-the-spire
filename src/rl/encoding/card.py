from typing import TypeAlias

import torch

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


def _encode_view_card(view_card: ViewCard, device: torch.device) -> torch.Tensor:
    upgraded = view_card.name.endswith("+")  # TODO: add `upgraded` field
    encoding = [0] * len(_EFFECT_KEY_MAX) + [
        view_card.cost / _COST_MAX,
        upgraded,
        view_card.requires_target,
        view_card.exhaust,
    ]
    for effect in view_card.effects:
        effect_key = _get_effect_key(effect)
        effect_key_pos = _EFFECT_KEY_POS[effect_key]

        # Get effect value
        effect_value = effect.value
        if effect_value is None:
            effect_value = 1.0

        encoding[effect_key_pos] = effect_value / _EFFECT_KEY_MAX[effect_key]

    return torch.tensor(encoding, dtype=torch.float32, device=device)


def _get_view_card_dummy() -> ViewCard:
    return ViewCard(
        "Dummy", CardColor.GREEN, CardType.ATTACK, CardRarity.BASIC, 1, [], False, False, False
    )


def get_encoding_card_dim() -> int:
    view_card_dummy = _get_view_card_dummy()
    encoding_card_dummy = _encode_view_card(view_card_dummy, torch.device("cpu"))
    return encoding_card_dummy.shape[0]


def encode_view_cards(
    view_cards: list[ViewCard], max_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view_cards_len = len(view_cards)
    if view_cards_len > max_size:
        raise ValueError(
            f"Length of card views ({view_cards_len}) exceeds maximum length ({max_size})"
        )

    mask_pad = torch.arange(max_size, dtype=torch.float32, device=device) < view_cards_len
    mask_active = torch.zeros(max_size, dtype=torch.float32, device=device)
    if not view_cards:
        # Get card encoding dimension
        encoding_card_dim = get_encoding_card_dim()

        # Return all-zeros tensor of shape (`max_size`, `card_encoding_dim`) and all-zeros mask_active
        return (
            torch.zeros(max_size, encoding_card_dim, dtype=torch.float32, device=device),
            mask_pad,
            mask_active,
        )

    encoding_cards = None
    for idx, view_card in enumerate(view_cards):
        # Get encoding
        encoding_card = _encode_view_card(view_card, device)

        if encoding_cards is None:
            # Intialize all-zeros tensor to hold all encodings, now that we now the enc. dimension
            encoding_cards = torch.zeros(
                max_size, encoding_card.shape[0], dtype=torch.float32, device=device
            )

        # Assign encoding
        encoding_cards[idx] = encoding_card

        # Set active mask index
        if view_card.is_active:
            if torch.sum(mask_active) > 1e-10:
                raise ValueError("Received more than one active `ViewCard`")

            mask_active[idx] = 1.0

    return encoding_cards, mask_pad, mask_active
