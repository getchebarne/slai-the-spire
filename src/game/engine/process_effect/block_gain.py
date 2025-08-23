from src.game.core.effect import Effect
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


BLOCK_MAX = 999


# TODO: frail
def process_effect_block_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Apply dexterity
    if ModifierType.DEXTERITY in target.modifier_map:
        stacks_current = target.modifier_map[ModifierType.DEXTERITY].stacks_current
        value += stacks_current

    target.block_current = min(target.block_current + value, BLOCK_MAX)

    return [], []
