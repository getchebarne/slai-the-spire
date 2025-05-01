from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_modifier_tick(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    for modifier_type, modifier_data in list(target.modifier_map.items()):
        if modifier_data.stacks_duration:
            modifier_data.stacks_current -= 1

            if modifier_data.stacks_current < modifier_data.stacks_min:
                del target.modifier_map[modifier_type]

    return [], []
