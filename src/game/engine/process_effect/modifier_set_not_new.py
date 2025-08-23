from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_modifier_set_not_new(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    for id_monster in entity_manager.id_monsters:
        monster = entity_manager.entities[id_monster]
        for modifier_data in monster.modifier_map.values():
            modifier_data.is_new = False

    character = entity_manager.entities[entity_manager.id_character]
    for modifier_data in character.modifier_map.values():
        modifier_data.is_new = False

    return [], []
