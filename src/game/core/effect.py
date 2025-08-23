from dataclasses import dataclass
from enum import Enum


class EffectType(Enum):
    # Add cards to hand
    ADD_TO_HAND_SHIV = "ADD_TO_HAND_SHIV"

    # Block manipulation
    BLOCK_GAIN = "BLOCK_GAIN"
    BLOCK_SET = "BLOCK_SET"

    # Calculated gamble
    CALCULATED_GAMBLE = "CALCULATED_GAMBLE"

    # Everything related to cards
    CARD_ACTIVE_SET = "CARD_ACTIVE_SET"
    CARD_ACTIVE_CLEAR = "CARD_ACTIVE_CLEAR"
    CARD_DISCARD = "CARD_DISCARD"
    CARD_DRAW = "CARD_DRAW"
    CARD_EXHAUST = "CARD_EXHAUST"
    CARD_PLAY = "CARD_PLAY"
    CARD_REMOVE = "CARD_REMOVE"
    CARD_REWARD_ROLL = "CARD_REWARD_ROLL"
    CARD_REWARD_SELECT = "CARD_REWARD_SELECT"
    CARD_UPGRADE = "CARD_UPGRADE"

    # Combat end and start
    COMBAT_START = "COMBAT_START"
    COMBAT_END = "COMBAT_END"

    # Deal damage
    DAMAGE_DEAL = "DAMAGE_DEAL"
    DAMAGE_DEAL_PHYSICAL = "DAMAGE_DEAL_PHYSICAL"

    # Death
    DEATH = "DEATH"

    # Energy manipulation
    ENERGY_GAIN = "ENERGY_GAIN"
    ENERGY_LOSS = "ENERGY_LOSS"

    # Game end - this effect is never processed, it's used to signal the game's end
    GAME_END = "GAME_END"

    # Health manipulation
    HEALTH_GAIN = "HEALTH_GAIN"
    HEALTH_LOSS = "HEALTH_LOSS"

    # Map node selection
    MAP_NODE_ACTIVE_SET = "MAP_NODE_ACTIVE_SET"

    # Everything related to manipulating modifiers
    MODIFIER_ACCURACY_GAIN = "MODIFIER_ACCURACY_GAIN"
    MODIFIER_AFTER_IMAGE_GAIN = "MODIFIER_AFTER_IMAGE_GAIN"
    MODIFIER_BLUR_GAIN = "MODIFIER_BLUR_GAIN"
    MODIFIER_BURST_GAIN = "MODIFIER_BURST_GAIN"
    MODIFIER_BURST_LOSS = "MODIFIER_BURST_LOSS"
    MODIFIER_DEXTERITY_GAIN = "MODIFIER_DEXTERITY_GAIN"
    MODIFIER_DOUBLE_DAMAGE_GAIN = "MODIFIER_DOUBLE_DAMAGE_GAIN"
    MODIFIER_INFINITE_BLADES_GAIN = "MODIFIER_INFINITE_BLADES_GAIN"
    MODIFIER_MODE_SHIFT_GAIN = "MODIFIER_MODE_SHIFT_GAIN"
    MODIFIER_NEXT_TURN_BLOCK_GAIN = "MODIFIER_NEXT_TURN_BLOCK_GAIN"
    MODIFIER_NEXT_TURN_ENERGY_GAIN = "MODIFIER_NEXT_TURN_ENERGY_GAIN"
    MODIFIER_PHANTASMAL_GAIN = "MODIFIER_PHANTASMAL_GAIN"
    MODIFIER_THOUSAND_CUTS_GAIN = "MODIFIER_THOUSAND_CUTS_GAIN"
    MODIFIER_TICK = "MODIFIER_TICK"
    MODIFIER_SET_NOT_NEW = "MODIFIER_SET_NOT_NEW"
    MODIFIER_RITUAL_GAIN = "MODIFIER_RITUAL_GAIN"
    MODIFIER_SHARP_HIDE_GAIN = "MODIFIER_SHARP_HIDE_GAIN"
    MODIFIER_SHARP_HIDE_LOSS = "MODIFIER_SHARP_HIDE_LOSS"
    MODIFIER_STRENGTH_GAIN = "MODIFIER_STRENGTH_GAIN"
    MODIFIER_VULNERABLE_GAIN = "MODIFIER_VULNERABLE_GAIN"
    MODIFIER_WEAK_GAIN = "MODIFIER_WEAK_GAIN"

    # Update monster's move
    MONSTER_MOVE_UPDATE = "MONSTER_MOVE_UPDATE"

    # Room enter
    ROOM_ENTER = "ROOM_ENTER"

    # Card target setting / clearing
    TARGET_CARD_SET = "TARGET_CARD_SET"
    TARGET_CARD_CLEAR = "TARGET_CARD_CLEAR"

    # Turn end / start
    TURN_END = "TURN_END"
    TURN_START = "TURN_START"


class EffectTargetType(Enum):
    CARD_IN_HAND = "CARD_IN_HAND"
    CARD_REWARD = "CARD_REWARD"
    CARD_TARGET = "CARD_TARGET"
    CHARACTER = "CHARACTER"
    MAP_NODE = "MAP_NODE"
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

    # Can be `None`, some effects are not created by entities but by the engine itself
    id_source: int | None = None

    # Effects can also be created with an `id_target`, in that case, `target_type` should be
    # `None`, as it doesn't have to be resolved. If `id_target` and `target_type` are both set,
    # `target_type` will be ignored (see `src.game.engine.effect_queue.py`)
    id_target: int | None = None
