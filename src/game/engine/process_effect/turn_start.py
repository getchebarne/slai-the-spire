from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager


def process_effect_turn_start(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    effects = []

    # Common effects
    block = 0
    if ModifierType.BLUR in target.modifier_map:
        block += target.block_current

    # Apply next turn block modifier. Must be applied after block reset
    if ModifierType.NEXT_TURN_BLOCK in target.modifier_map:
        stacks_current = target.modifier_map[ModifierType.NEXT_TURN_BLOCK].stacks_current
        block += stacks_current

        # Clear modifier
        del target.modifier_map[ModifierType.NEXT_TURN_BLOCK]

    effects.append(Effect(EffectType.BLOCK_SET, block, target=target))

    # Apply phantasmal
    if ModifierType.PHANTASMAL in target.modifier_map:
        effects.append(Effect(EffectType.MODIFIER_DOUBLE_DAMAGE_GAIN, 1, target=target))

    # Character-specific effects
    if isinstance(target, EntityCharacter):
        effects += [
            Effect(EffectType.CARD_DRAW, 5),
            Effect(
                EffectType.ENERGY_GAIN, entity_manager.energy.max - entity_manager.energy.current
            ),
            Effect(EffectType.MODIFIER_TICK, target=entity_manager.character),
        ]
        effects += [
            Effect(EffectType.MODIFIER_TICK, target=monster) for monster in entity_manager.monsters
        ]

        # Apply next turn energy modifier
        if ModifierType.NEXT_TURN_ENERGY in target.modifier_map:
            stacks_current = target.modifier_map[ModifierType.NEXT_TURN_ENERGY].stacks_current
            effects += [Effect(EffectType.ENERGY_GAIN, stacks_current)]

            # Clear modifier
            del target.modifier_map[ModifierType.NEXT_TURN_ENERGY]

        # Infinite blades
        if ModifierType.INFINITE_BLADES in target.modifier_map:
            stacks_current = target.modifier_map[ModifierType.INFINITE_BLADES].stacks_current
            effects += [Effect(EffectType.ADD_TO_HAND_SHIV, stacks_current)]

    return [], effects
