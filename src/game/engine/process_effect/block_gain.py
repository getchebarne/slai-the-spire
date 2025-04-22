from src.game.combat.effect import Effect
from src.game.entity.manager import EntityManager


BLOCK_MAX = 999


def process_effect_block_gain(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]

    target.block_current = min(target.block_current + effect.value, BLOCK_MAX)
    return [], []
