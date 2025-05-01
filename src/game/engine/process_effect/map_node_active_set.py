from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_map_node_active_set(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    entity_manager.id_map_node_active = id_target

    # TODO: Maw Bank
    return [], [Effect(EffectType.ROOM_ENTER)]
