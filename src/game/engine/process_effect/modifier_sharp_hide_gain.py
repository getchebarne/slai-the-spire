from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def process_effect_modifier_sharp_hide_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]
    if ModifierType.SHARP_HIDE in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.SHARP_HIDE]
        modifier_data.stacks_current = min(modifier_data.stacks_current + value, STACKS_MAX)

        return [], []

    target.modifier_map[ModifierType.SHARP_HIDE] = ModifierData(
        min(value, STACKS_MAX), STACKS_MIN, STACKS_MAX, STACKS_DURATION
    )

    return [], []
