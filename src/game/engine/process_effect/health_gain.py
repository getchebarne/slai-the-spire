from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_health_gain(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]

    target.health_current = min(target.health_max, target.health_current + effect.value)

    return [], []
