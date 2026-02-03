"""
Consolidated modifier gain handlers.

Most modifiers follow the same pattern: add stacks if exists, create if not.
This module provides a generic implementation with configuration-driven behavior.
"""

from functools import partial

from src.game.core.effect import Effect
from src.game.entity.actor import ModifierConfig
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


# Configuration for each modifier type
MODIFIER_CONFIGS: dict[ModifierType, ModifierConfig] = {
    # Buffs (permanent stacks)
    ModifierType.ACCURACY: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.AFTER_IMAGE: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.BLUR: ModifierConfig(is_buff=True, stacks_duration=True),
    ModifierType.BURST: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.DEXTERITY: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.DOUBLE_DAMAGE: ModifierConfig(is_buff=True, stacks_duration=True),
    ModifierType.INFINITE_BLADES: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.NEXT_TURN_BLOCK: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.NEXT_TURN_ENERGY: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.PHANTASMAL: ModifierConfig(is_buff=True, stacks_duration=True),
    ModifierType.RITUAL: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.SHARP_HIDE: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.STRENGTH: ModifierConfig(is_buff=True, stacks_duration=False),
    ModifierType.THOUSAND_CUTS: ModifierConfig(is_buff=True, stacks_duration=False),
    # Debuffs (duration-based)
    ModifierType.VULNERABLE: ModifierConfig(is_buff=False, stacks_duration=True),
    ModifierType.WEAK: ModifierConfig(is_buff=False, stacks_duration=True),
}


def _process_modifier_gain(
    modifier_type: ModifierType,
    entity_manager: EntityManager,
    **kwargs,
) -> tuple[list[Effect], list[Effect]]:
    """Generic modifier gain handler."""
    config = MODIFIER_CONFIGS[modifier_type]
    value = kwargs["value"]
    target = kwargs["target"]

    if modifier_type in target.modifier_map:
        data = target.modifier_map[modifier_type]
        data.stacks_current = min(data.stacks_current + value, config.stacks_max)
    else:
        target.modifier_map[modifier_type] = ModifierData(
            config=config,
            is_new=True,
            stacks_current=min(value, config.stacks_max),
        )

    return [], []


# Export specific handlers via partial application
process_effect_modifier_accuracy_gain = partial(_process_modifier_gain, ModifierType.ACCURACY)
process_effect_modifier_after_image_gain = partial(
    _process_modifier_gain, ModifierType.AFTER_IMAGE
)
process_effect_modifier_blur_gain = partial(_process_modifier_gain, ModifierType.BLUR)
process_effect_modifier_burst_gain = partial(_process_modifier_gain, ModifierType.BURST)
process_effect_modifier_dexterity_gain = partial(_process_modifier_gain, ModifierType.DEXTERITY)
process_effect_modifier_double_damage_gain = partial(
    _process_modifier_gain, ModifierType.DOUBLE_DAMAGE
)
process_effect_modifier_infinite_blades_gain = partial(
    _process_modifier_gain, ModifierType.INFINITE_BLADES
)
process_effect_modifier_next_turn_block_gain = partial(
    _process_modifier_gain, ModifierType.NEXT_TURN_BLOCK
)
process_effect_modifier_next_turn_energy_gain = partial(
    _process_modifier_gain, ModifierType.NEXT_TURN_ENERGY
)
process_effect_modifier_phantasmal_gain = partial(_process_modifier_gain, ModifierType.PHANTASMAL)
process_effect_modifier_ritual_gain = partial(_process_modifier_gain, ModifierType.RITUAL)
process_effect_modifier_sharp_hide_gain = partial(_process_modifier_gain, ModifierType.SHARP_HIDE)
process_effect_modifier_strength_gain = partial(_process_modifier_gain, ModifierType.STRENGTH)
process_effect_modifier_thousand_cuts_gain = partial(
    _process_modifier_gain, ModifierType.THOUSAND_CUTS
)
process_effect_modifier_vulnerable_gain = partial(_process_modifier_gain, ModifierType.VULNERABLE)
process_effect_modifier_weak_gain = partial(_process_modifier_gain, ModifierType.WEAK)
