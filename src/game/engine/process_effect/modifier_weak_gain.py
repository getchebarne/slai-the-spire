from src.game.combat.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = True


def process_effect_modifier_weak_gain(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    target = entity_manager.entities[effect.id_target]
    if ModifierType.WEAK in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.WEAK]
        modifier_data.stacks_current = min(modifier_data.stacks_current + effect.value, STACKS_MAX)

        return [], []

    target.modifier_map[ModifierType.WEAK] = ModifierData(
        min(effect.value, STACKS_MAX), STACKS_MIN, STACKS_MAX, STACKS_DURATION
    )

    return [], []
