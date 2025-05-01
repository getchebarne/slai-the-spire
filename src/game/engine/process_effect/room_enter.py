import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager
from src.game.entity.map_node import RoomType
from src.game.level.exordium.combat_cultist import set_level_exoridium_combat_cultist
from src.game.level.exordium.combat_fungi_beast_two import \
    set_level_exoridium_combat_fungi_beast_two
from src.game.level.exordium.combat_jaw_worm import set_level_exoridium_combat_jaw_worm


def process_effect_room_enter(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    map_node_active = entity_manager.entities[entity_manager.id_map_node_active]

    if map_node_active.room_type == RoomType.COMBAT_MONSTER:
        set_level_fn = random.choice(
            [
                set_level_exoridium_combat_cultist,
                set_level_exoridium_combat_fungi_beast_two,
                set_level_exoridium_combat_jaw_worm,
            ]
        )
        set_level_fn(entity_manager)

        return [], [Effect(EffectType.COMBAT_START)]

    if map_node_active.room_type == RoomType.REST_SITE:
        return [], []

    raise ValueError("TODO: add message")
