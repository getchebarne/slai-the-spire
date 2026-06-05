import numpy as np
import torch
from slai import Card
from slai import CardColor
from slai import CardKind
from slai import CardName
from slai import CardRarity


# Enums
_MAP_CARD_NAME = {card_name: i for i, card_name in enumerate(CardName)}
_MAP_CARD_KIND = {card_kind: i for i, card_kind in enumerate(CardKind)}
_MAP_CARD_COLOR = {card_color: i for i, card_color in enumerate(CardColor)}
_MAP_CARD_RARITY = {card_rarity: i for i, card_rarity in enumerate(CardRarity)}

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
    + _COST_VECTOR_DIM  # Cost vector
    + 1  # Cost scalar
    + 1  # Upgraded
    + 1  # Exhaust
    + 1  # Innate
    + 1  # Ethereal
    + 1  # Retain
    + 1  # Requires target
    + 1  # Playable
)


# TODO: encode effects
def encode_card_into(card: Card, pos: int, out: np.ndarray) -> int:
    # Name OHE
    idx = _MAP_CARD_NAME.get(card.name)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_MAP_CARD_NAME)

    # Kind OHE
    idx = _MAP_CARD_KIND.get(card.kind)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_MAP_CARD_KIND)

    # Color OHE
    idx = _MAP_CARD_COLOR.get(card.color)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_MAP_CARD_COLOR)

    # Rarity OHE
    idx = _MAP_CARD_RARITY.get(card.rarity)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_MAP_CARD_RARITY)

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
    pos += 8

    return pos


def encode_batch_cards(
    batch_cards: list[list[Card]], max_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(batch_cards)

    # Allocate arrays
    x_out = np.zeros((batch_size, max_size, ENCODING_DIM_CARD), dtype=np.float32)
    x_pad = np.zeros((batch_size, max_size), dtype=bool)

    for b, cards in enumerate(batch_cards):
        cards = cards[:max_size]
        for i, card in enumerate(cards):
            encode_card_into(card, 0, x_out[b, i])
            x_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
    )
