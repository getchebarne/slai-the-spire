from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


ENERGY_MAX = 999


def process_effect_energy_gain(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    energy = entity_manager.entities[entity_manager.id_energy]
    energy.current = min(energy.current + effect.value, ENERGY_MAX)

    return [], []
