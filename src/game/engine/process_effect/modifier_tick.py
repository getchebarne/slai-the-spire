from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_modifier_tick(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    for modifier_type, modifier_data in list(target.modifier_map.items()):
        if modifier_data.config.stacks_duration and not modifier_data.is_new:
            modifier_data.stacks_current -= 1

            if modifier_data.stacks_current < modifier_data.config.stacks_min:
                del target.modifier_map[modifier_type]

    return [], []
