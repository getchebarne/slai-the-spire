from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


IS_BUFF = False
STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = True


def process_effect_modifier_weak_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    target = kwargs["target"]

    if ModifierType.WEAK in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.WEAK]
        modifier_data.stacks_current = min(modifier_data.stacks_current + value, STACKS_MAX)

        return [], []

    target.modifier_map[ModifierType.WEAK] = ModifierData(
        IS_BUFF,
        True,
        min(value, STACKS_MAX),
        STACKS_MIN,
        STACKS_MAX,
        STACKS_DURATION,
    )

    return [], []
