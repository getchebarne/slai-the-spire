from dataclasses import dataclass
from enum import Enum


EFFECT_VALUE_PLACEHOLDER_MODIFIER_DATA_CURRENT_STACKS = -1


class EffectType(Enum):
    DEAL_DAMAGE = "DEAL_DAMAGE"
    DECREASE_ENERGY = "DECREASE_ENERGY"
    DISCARD = "DISCARD"
    DRAW_CARD = "DRAW_CARD"
    END_TURN = "END_TURN"
    GAIN_BLOCK = "GAIN_BLOCK"
    GAIN_STRENGTH = "GAIN_STRENGTH"
    GAIN_WEAK = "GAIN_WEAK"
    MOD_TICK = "MOD_TICK"  # TODO: rename
    PLAY_CARD = "PLAY_CARD"
    REFILL_ENERGY = "REFILL_ENERGY"
    SHUFFLE_DECK_INTO_DRAW_PILE = "SHUFFLE_DECK_INTO_DRAW_PILE"
    UPDATE_MOVE = "UPDATE_MOVE"
    ZERO_BLOCK = "ZERO_BLOCK"
    LOSE_HP = "LOSE_HP"
    DEATH = "DEATH"
    GAIN_VULNERABLE = "GAIN_VULNERABLE"

    # Target setting / clearing
    TARGET_EFFECT_SET = "TARGET_EFFECT_SET"
    TARGET_EFFECT_CLEAR = "TARGET_EFFECT_CLEAR"
    TARGET_CARD_SET = "TARGET_CARD_SET"
    TARGET_CARD_CLEAR = "TARGET_CARD_CLEAR"

    # Active card setting / clearing
    CARD_ACTIVE_SET = "CARD_ACTIVE_SET"
    CARD_ACTIVE_CLEAR = "CARD_ACTIVE_CLEAR"


class EffectTargetType(Enum):
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
