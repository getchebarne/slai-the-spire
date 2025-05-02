from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_energy_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    energy = entity_manager.entities[entity_manager.id_energy]
    if energy.current < value:
        raise ValueError(f"Can't dercrease current energy ({energy.current}) by {value}")

    energy.current = energy.current - value

    return [], []
