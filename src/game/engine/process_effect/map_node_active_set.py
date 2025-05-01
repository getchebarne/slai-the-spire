from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_map_node_active_set(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    entity_manager.id_map_node_active = effect.id_target

    # TODO: Maw Bank
    return [], [Effect(EffectType.ROOM_ENTER)]
