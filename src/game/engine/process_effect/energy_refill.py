from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_energy_refill(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    energy = entity_manager.entities[entity_manager.id_energy]
    energy.current = energy.max

    return [], []
