from src.game.core.effect import Effect
from src.game.core.effect import EffectType
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
        energy = entity_manager.energy
        effects += [
            Effect(EffectType.CARD_DRAW, 5),
            Effect(EffectType.ENERGY_GAIN, energy.max - energy.current),
        ] + [
            Effect(EffectType.MODIFIER_TICK, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ]

    elif isinstance(target, EntityMonster):
        effects.append(Effect(EffectType.MODIFIER_TICK, id_target=entity_manager.id_character))

    return [], effects
