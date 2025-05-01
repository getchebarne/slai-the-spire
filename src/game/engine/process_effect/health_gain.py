from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_health_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    target.health_current = min(target.health_max, target.health_current + value)

    return [], []
