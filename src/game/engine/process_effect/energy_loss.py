from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_energy_loss(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    energy = entity_manager.entities[entity_manager.id_energy]

    if energy.current < effect.value:
        raise ValueError(f"Can't dercrease current energy ({energy.current}) by {effect.value}")

    energy.current = energy.current - effect.value

    return [], []
