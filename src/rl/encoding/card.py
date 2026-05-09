"""Card encoder.

Pre-migration this module introspected the old emulator's card factories
to derive a per-card-name one-hot and a per-EffectKey value-aware encoding.
slai exposes neither a "list of all cards" API nor a stable Python-side
view of effect values, so we use a structural encoding instead:

  - One-hot of `CardKind` (Attack / Skill / Power / Curse / Status)
  - One-hot of `CardColor` (Green / Colorless / Curse)
  - One-hot of `CardRarity` (Basic / Common / Uncommon / Rare / Special / Curse)
  - Cost as scalar + one-hot bucket
  - Boolean flags: upgraded, exhaust, innate, ethereal, retain, requires_target, playable
  - Effect histogram: count of each `Effect.*` variant on the card
  - Target histogram: count of each `CandidatePool` referenced by effects' targets
  - Selection histogram: count of each `Selection` kind referenced by effects' targets

Per-card-name one-hot can be added back by hardcoding slai's card roster
(see `slai/src/cards/`) once we want to feed the model identity bits.
"""

import math
from enum import Enum

import numpy as np
import slai
import torch

from src.rl.constants import MAX_SIZE_COMBAT_CARD_REWARD
from src.rl.constants import MAX_SIZE_DECK
from src.rl.constants import MAX_SIZE_DISC_PILE
from src.rl.constants import MAX_SIZE_DRAW_PILE
from src.rl.constants import MAX_SIZE_HAND


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


# ---------- Snapshot slai enum surfaces at module load ----------

_CARD_KIND_NAMES = sorted(n for n in dir(slai.CardKind) if not n.startswith("_"))
_CARD_KIND_TO_IDX = {getattr(slai.CardKind, n): i for i, n in enumerate(_CARD_KIND_NAMES)}

_CARD_COLOR_NAMES = sorted(n for n in dir(slai.CardColor) if not n.startswith("_"))
_CARD_COLOR_TO_IDX = {getattr(slai.CardColor, n): i for i, n in enumerate(_CARD_COLOR_NAMES)}

_CARD_RARITY_NAMES = sorted(n for n in dir(slai.CardRarity) if not n.startswith("_"))
_CARD_RARITY_TO_IDX = {getattr(slai.CardRarity, n): i for i, n in enumerate(_CARD_RARITY_NAMES)}

_EFFECT_NAMES = sorted(n for n in dir(slai.Effect) if not n.startswith("_"))
_EFFECT_CLASSES: list[type] = [getattr(slai.Effect, n) for n in _EFFECT_NAMES]

_CANDIDATE_POOL_NAMES = sorted(n for n in dir(slai.CandidatePool) if not n.startswith("_"))
_CANDIDATE_POOL_TO_IDX = {
    getattr(slai.CandidatePool, n): i for i, n in enumerate(_CANDIDATE_POOL_NAMES)
}

_SELECTION_NAMES = sorted(n for n in dir(slai.Selection) if not n.startswith("_"))
_SELECTION_CLASSES: list[type] = [getattr(slai.Selection, n) for n in _SELECTION_NAMES]


# Cost normalization
_COST_MAX = 5  # X-cost cards top out around energy.max
_COST_SQRT_MIN = 0
_COST_SQRT_MAX = int(math.sqrt(_COST_MAX))
_COST_SQRT_DIM = _COST_SQRT_MAX - _COST_SQRT_MIN + 1

# Number of boolean flag scalars per card
_NUM_FLAG_SCALARS = 7  # upgraded, exhaust, innate, ethereal, retain, requires_target, playable


def get_encoding_dim_card() -> int:
    return (
        len(_CARD_KIND_NAMES)
        + len(_CARD_COLOR_NAMES)
        + len(_CARD_RARITY_NAMES)
        + _COST_SQRT_DIM
        + 1  # cost scalar
        + _NUM_FLAG_SCALARS
        + len(_EFFECT_NAMES)  # effect-kind histogram
        + len(_CANDIDATE_POOL_NAMES)  # target-pool histogram
        + len(_SELECTION_NAMES)  # selection histogram
    )


_ENCODING_DIM_CARD = get_encoding_dim_card()


def _encode_view_card_into(out: np.ndarray, view_card: slai.Card) -> None:
    """Encode a slai.Card into a pre-allocated numpy array."""
    pos = 0

    # CardKind one-hot
    idx = _CARD_KIND_TO_IDX.get(view_card.kind)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_KIND_NAMES)

    # CardColor one-hot
    idx = _CARD_COLOR_TO_IDX.get(view_card.color)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_COLOR_NAMES)

    # CardRarity one-hot
    idx = _CARD_RARITY_TO_IDX.get(view_card.rarity)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_RARITY_NAMES)

    # Cost sqrt one-hot
    cost = max(0, min(view_card.cost, _COST_MAX))
    cost_sqrt = max(_COST_SQRT_MIN, min(int(math.sqrt(cost)), _COST_SQRT_MAX))
    out[pos + cost_sqrt - _COST_SQRT_MIN] = 1.0
    pos += _COST_SQRT_DIM

    # Cost scalar
    out[pos] = cost / _COST_MAX
    pos += 1

    # Flag scalars
    out[pos] = float(view_card.upgraded)
    out[pos + 1] = float(view_card.exhaust)
    out[pos + 2] = float(view_card.innate)
    out[pos + 3] = float(view_card.ethereal)
    out[pos + 4] = float(view_card.retain)
    out[pos + 5] = float(view_card.requires_target)
    out[pos + 6] = float(view_card.playable)
    pos += _NUM_FLAG_SCALARS

    # Effect histogram + target pool / selection histogram (over the card's effects)
    for effect in view_card.effects:
        for idx, eff_cls in enumerate(_EFFECT_CLASSES):
            if isinstance(effect, eff_cls):
                out[pos + idx] += 1.0
                break
        target = getattr(effect, "target", None)
        if target is not None:
            cand_idx = _CANDIDATE_POOL_TO_IDX.get(target.candidates)
            if cand_idx is not None:
                out[pos + len(_EFFECT_NAMES) + cand_idx] += 1.0
            for sel_idx, sel_cls in enumerate(_SELECTION_CLASSES):
                if isinstance(target.selection, sel_cls):
                    out[
                        pos + len(_EFFECT_NAMES) + len(_CANDIDATE_POOL_NAMES) + sel_idx
                    ] += 1.0
                    break


def encode_batch_view_cards(
    batch_view_cards: list[list[slai.Card]], card_pile: CardPile, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch of card lists using NumPy pre-allocation."""
    max_size = _CARD_PILE_TO_MAX_SIZE[card_pile]
    batch_size = len(batch_view_cards)

    x_out = np.zeros((batch_size, max_size, _ENCODING_DIM_CARD), dtype=np.float32)
    x_mask_pad = np.zeros((batch_size, max_size), dtype=bool)

    for b, view_cards in enumerate(batch_view_cards):
        view_cards = view_cards[:max_size]
        for i, view_card in enumerate(view_cards):
            _encode_view_card_into(x_out[b, i], view_card)
            x_mask_pad[b, i] = True

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_mask_pad).to(device),
    )
