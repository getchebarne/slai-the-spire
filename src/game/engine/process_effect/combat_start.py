from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_combat_start(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    return [], [
        Effect(EffectType.CARD_SHUFFLE_DECK_INTO_DRAW_PILE),
        *[
            Effect(EffectType.MONSTER_MOVE_UPDATE, id_target=id_monster)
            for id_monster in entity_manager.id_monsters
        ],
    ] + [Effect(EffectType.TURN_START, id_target=entity_manager.id_character)]
