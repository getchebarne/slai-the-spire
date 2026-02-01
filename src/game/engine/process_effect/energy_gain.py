from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


ENERGY_MAX = 999


def process_effect_energy_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    entity_manager.energy.current = min(entity_manager.energy.current + value, ENERGY_MAX)

    return [], []
