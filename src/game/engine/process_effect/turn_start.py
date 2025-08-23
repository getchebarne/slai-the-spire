from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.character import EntityCharacter
from src.game.entity.manager import EntityManager
from src.game.entity.monster import EntityMonster


def process_effect_turn_start(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Common effects
    effects = [Effect(EffectType.BLOCK_RESET, id_target=id_target)]

    # Character-specific effects
    if isinstance(target, EntityCharacter):
        energy = entity_manager.entities[entity_manager.id_energy]
        effects += [
            Effect(EffectType.CARD_DRAW, 5),
            Effect(EffectType.ENERGY_GAIN, energy.max - energy.current),
        ] + [
            Effect(EffectType.MODIFIER_TICK, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ]
        # Apply next turn energy modifier
        if ModifierType.NEXT_TURN_ENERGY in target.modifier_map:
            stacks_current = target.modifier_map[ModifierType.NEXT_TURN_ENERGY].stacks_current
            effects += [Effect(EffectType.ENERGY_GAIN, stacks_current)]

            # Clear modifier
            del target.modifier_map[ModifierType.NEXT_TURN_ENERGY]

    elif isinstance(target, EntityMonster):
        effects.append(Effect(EffectType.MODIFIER_TICK, id_target=entity_manager.id_character))

    # Apply next turn block modifier. Must be applied after block reset
    if ModifierType.NEXT_TURN_BLOCK in target.modifier_map:
        stacks_current = target.modifier_map[ModifierType.NEXT_TURN_BLOCK].stacks_current
        effects += [
            Effect(
                EffectType.BLOCK_GAIN, stacks_current, EffectTargetType.SOURCE, id_source=id_target
            )
        ]

        # Clear modifier
        del target.modifier_map[ModifierType.NEXT_TURN_BLOCK]

    return [], effects
