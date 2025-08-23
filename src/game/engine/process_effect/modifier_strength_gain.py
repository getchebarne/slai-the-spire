from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


IS_BUFF = True
STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def process_effect_modifier_strength_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    if ModifierType.STRENGTH in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.STRENGTH]
        modifier_data.stacks_current = min(modifier_data.stacks_current + value, STACKS_MAX)

        return [], []

    target.modifier_map[ModifierType.STRENGTH] = ModifierData(
        IS_BUFF,
        True,
        min(value, STACKS_MAX),
        STACKS_MIN,
        STACKS_MAX,
        STACKS_DURATION,
    )

    return [], []
