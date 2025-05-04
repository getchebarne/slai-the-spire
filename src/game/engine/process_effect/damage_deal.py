from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_damage_deal(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Calculate damage over block
    damage_over_block = max(0, value - target.block_current)

    # Remove block
    target.block_current = max(0, target.block_current - value)

    if damage_over_block > 0:
        # Return a top effect to subtract the damage over block from the target's current health
        return [], [Effect(EffectType.HEALTH_LOSS, value=damage_over_block, id_target=id_target)]

    return [], []
