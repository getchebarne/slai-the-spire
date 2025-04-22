from dataclasses import replace

from src.game.combat.phase import get_end_of_turn_effects
from src.game.combat.phase import get_start_of_turn_effects
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


BLOCK_MAX = 999


def process_effect_end_turn(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    # Character's turn end
    effects = get_end_of_turn_effects(entity_manager, entity_manager.id_character)

    # Monsters
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]

        # Turn start
        effects += get_start_of_turn_effects(entity_manager, id_monster)

        # Move's effects
        effects += [
            replace(effect, id_source=id_monster)
            for effect in monster.move_map[monster.move_name_current]
        ]

        # Update move
        effects += [Effect(EffectType.MONSTER_MOVE_UPDATE, id_target=id_monster)]

        # Turn end
        effects += get_end_of_turn_effects(entity_manager, id_monster)

    # Character's turn start
    effects += get_start_of_turn_effects(entity_manager, entity_manager.id_character)

    return effects, []
