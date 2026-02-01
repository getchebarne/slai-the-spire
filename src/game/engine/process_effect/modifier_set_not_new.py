from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_modifier_set_not_new(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    for monster in entity_manager.monsters:
        for modifier_data in monster.modifier_map.values():
            modifier_data.is_new = False

    for modifier_data in entity_manager.character.modifier_map.values():
        modifier_data.is_new = False

    return [], []
