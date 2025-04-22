from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def process_effect_modifier_strength_gain(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]

    if ModifierType.STRENGTH in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.STRENGTH]
        modifier_data.stacks_current = min(modifier_data.stacks_current + effect.value, STACKS_MAX)

        return [], []

    target.modifier_map[ModifierType.STRENGTH] = ModifierData(
        min(effect.value, STACKS_MAX), STACKS_MIN, STACKS_MAX, STACKS_DURATION
    )

    return [], []
