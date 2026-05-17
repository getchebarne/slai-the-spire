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
#
# CardKind / CardColor / CardRarity / RoomKind / ModifierKind / IntentKind /
# CandidatePool / CardName / MonsterName / RelicName / RelicTier / ActionType
# are real `enum.IntEnum` types in slai's Python layer (see slai's
# `_to_intenum` shim) — iterate the class directly.
#
# Effect / Selection / Phase / Target / CardCostKind are PyO3 complex enums
# (parent class + nested variant classes); iterate via dir() filtered by
# `isinstance(v, type)`.

_CARD_KIND_NAMES = [m.name for m in slai.CardKind]
_CARD_KIND_TO_IDX = {m: i for i, m in enumerate(slai.CardKind)}

_CARD_COLOR_NAMES = [m.name for m in slai.CardColor]
_CARD_COLOR_TO_IDX = {m: i for i, m in enumerate(slai.CardColor)}

_CARD_RARITY_NAMES = [m.name for m in slai.CardRarity]
_CARD_RARITY_TO_IDX = {m: i for i, m in enumerate(slai.CardRarity)}

_EFFECT_NAMES = sorted(n for n in dir(slai.Effect) if not n.startswith("_"))
_EFFECT_CLASSES: list[type] = [getattr(slai.Effect, n) for n in _EFFECT_NAMES]

_CANDIDATE_POOL_NAMES = [m.name for m in slai.CandidatePool]
_CANDIDATE_POOL_TO_IDX = {m: i for i, m in enumerate(slai.CandidatePool)}

_SELECTION_NAMES = sorted(n for n in dir(slai.SelectionKind) if not n.startswith("_"))
_SELECTION_CLASSES: list[type] = [getattr(slai.SelectionKind, n) for n in _SELECTION_NAMES]

# Per-card-name one-hot. IntEnum members iterate in declaration order
# (matches int discriminant) — that's a stable enumeration.
_CARD_NAMES = list(slai.CardName)
_CARD_NAME_TO_IDX = {n: i for i, n in enumerate(_CARD_NAMES)}
_NUM_CARD_NAMES = len(_CARD_NAMES)


# Cost normalization
_COST_MAX = 5  # X-cost cards top out around energy.max
_COST_SQRT_MIN = 0
_COST_SQRT_MAX = int(math.sqrt(_COST_MAX))
_COST_SQRT_DIM = _COST_SQRT_MAX - _COST_SQRT_MIN + 1

# Number of boolean flag scalars per card
_NUM_FLAG_SCALARS = 7  # upgraded, exhaust, innate, ethereal, retain, requires_target, playable


def get_encoding_dim_card() -> int:
    return (
        _NUM_CARD_NAMES  # per-card-name one-hot (78 today)
        + len(_CARD_KIND_NAMES)
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


# =============================================================================
# Card identity — single source of truth for the policy-relevant attribute
# list. Used by:
#   - _encode_view_card_into  (to build the per-slot encoder feature vector)
#   - _card_identity          (to compute the per-slot group identity used
#                              by grouped sampling — see heads.py)
# Adding an attribute the policy should respond to: add it here. The encoder
# destructures this tuple at its top, so forgetting to update one consumer
# raises at module load via a tuple-unpack length mismatch.
# =============================================================================


def _card_policy_features(card: slai.Card) -> tuple:
    """Tuple of card attributes the policy depends on. Stable per
    (card_name, upgraded) for template-derived attrs; varies per-instance for
    `cost` (X-cost / dynamic-cost / free_to_play_once), `retain` (settable
    via Well Laid Plans), and `playable` (Entangled etc.)."""
    return (
        card.name,
        card.kind,
        card.color,
        card.rarity,
        card.cost,
        card.upgraded,
        card.exhaust,
        card.innate,
        card.ethereal,
        card.retain,
        card.requires_target,
        card.playable,
        # Effects are template-derived from (card_name, upgraded) but listed
        # explicitly so the encoder/identity coupling stays honest if that
        # ever changes. slai effects are frozen pyclasses with eq/hash.
        tuple(card.effects),
    )


def card_identity_ids(cards: list[slai.Card]) -> list[int]:
    """Assign per-state integer group ids to a sequence of cards. Two cards
    with identical `_card_policy_features` get the same id. Ids are
    contiguous starting at 0. Use -1 for invalid/padded slots (the caller
    pads). No risk of hash collisions across cards within the same call."""
    seen: dict[tuple, int] = {}
    out: list[int] = []
    for card in cards:
        key = _card_policy_features(card)
        gid = seen.setdefault(key, len(seen))
        out.append(gid)
    return out


def _encode_view_card_into(out: np.ndarray, view_card: slai.Card) -> None:
    """Encode a slai.Card into a pre-allocated numpy array.

    Reads attributes via `_card_policy_features` so the encoder and the
    identity hash stay coupled — adding/removing an attribute touches one
    list and propagates here via the tuple unpack.
    """
    (
        card_name,
        kind,
        color,
        rarity,
        cost_raw,
        upgraded,
        exhaust,
        innate,
        ethereal,
        retain,
        requires_target,
        playable,
        effects,
    ) = _card_policy_features(view_card)

    pos = 0

    # Per-card-name one-hot
    idx = _CARD_NAME_TO_IDX.get(card_name)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += _NUM_CARD_NAMES

    # CardKind one-hot
    idx = _CARD_KIND_TO_IDX.get(kind)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_KIND_NAMES)

    # CardColor one-hot
    idx = _CARD_COLOR_TO_IDX.get(color)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_COLOR_NAMES)

    # CardRarity one-hot
    idx = _CARD_RARITY_TO_IDX.get(rarity)
    if idx is not None:
        out[pos + idx] = 1.0
    pos += len(_CARD_RARITY_NAMES)

    # Cost sqrt one-hot
    cost = max(0, min(cost_raw, _COST_MAX))
    cost_sqrt = max(_COST_SQRT_MIN, min(int(math.sqrt(cost)), _COST_SQRT_MAX))
    out[pos + cost_sqrt - _COST_SQRT_MIN] = 1.0
    pos += _COST_SQRT_DIM

    # Cost scalar
    out[pos] = cost / _COST_MAX
    pos += 1

    # Flag scalars
    out[pos] = float(upgraded)
    out[pos + 1] = float(exhaust)
    out[pos + 2] = float(innate)
    out[pos + 3] = float(ethereal)
    out[pos + 4] = float(retain)
    out[pos + 5] = float(requires_target)
    out[pos + 6] = float(playable)
    pos += _NUM_FLAG_SCALARS

    # Effect histogram + target pool / selection histogram (over the card's effects)
    for effect in effects:
        for idx, eff_cls in enumerate(_EFFECT_CLASSES):
            if isinstance(effect, eff_cls):
                out[pos + idx] += 1.0
                break
        target = getattr(effect, "target", None)
        if target is not None:
            cand_idx = _CANDIDATE_POOL_TO_IDX.get(target.candidate_pool)
            if cand_idx is not None:
                out[pos + len(_EFFECT_NAMES) + cand_idx] += 1.0
            for sel_idx, sel_cls in enumerate(_SELECTION_CLASSES):
                if isinstance(target.selection_kind, sel_cls):
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
