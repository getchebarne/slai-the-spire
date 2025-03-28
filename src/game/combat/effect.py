from dataclasses import dataclass
from enum import Enum


class EffectType(Enum):
    DEAL_DAMAGE = "DEAL_DAMAGE"
    DECREASE_ENERGY = "DECREASE_ENERGY"
    DISCARD = "DISCARD"
    DRAW_CARD = "DRAW_CARD"
    GAIN_BLOCK = "GAIN_BLOCK"
    GAIN_STR = "GAIN_STR"
    GAIN_WEAK = "GAIN_WEAK"
    MOD_TICK = "MOD_TICK"  # TODO: rename
    PLAY_CARD = "PLAY_CARD"
    REFILL_ENERGY = "REFILL_ENERGY"
    SHUFFLE_DECK_INTO_DRAW_PILE = "SHUFFLE_DECK_INTO_DRAW_PILE"
    UPDATE_MOVE = "UPDATE_MOVE"
    ZERO_BLOCK = "ZERO_BLOCK"


class EffectTargetType(Enum):
    CARD_ACTIVE = "CARD_ACTIVE"
    CARD_IN_HAND = "CARD_IN_HAND"
    CARD_TARGET = "CARD_TARGET"
    CHARACTER = "CHARACTER"
    MONSTER = "MONSTER"
    SOURCE = "SOURCE"


class EffectSelectionType(Enum):
    INPUT = "INPUT"
    RANDOM = "RANDOM"


@dataclass(frozen=True)
class Effect:
    type: EffectType
    value: int | None = None
    target_type: EffectTargetType | None = None
    selection_type: EffectSelectionType | None = None


@dataclass(frozen=True)
class SourcedEffect:
    effect: Effect
    id_source: int | None = None
    id_target: int | None = None
