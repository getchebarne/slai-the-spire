from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_map_node_active_set(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    entity_manager.map_node_active = target

    # Reset card rewards TODO: here?
    entity_manager.card_reward = []

    # TODO: Maw Bank
    return [], [Effect(EffectType.ROOM_ENTER)]
