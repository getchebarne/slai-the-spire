from src.game.core.effect import Effect
from src.game.entity.manager import EntityManager


def process_effect_energy_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    if entity_manager.energy.current < value:
        raise ValueError(
            f"Can't decrease current energy ({entity_manager.energy.current}) by {value}"
        )

    entity_manager.energy.current = entity_manager.energy.current - value

    return [], []
