from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


IS_BUFF = True
STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def process_effect_modifier_next_turn_energy_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]

    character = entity_manager.character

    if ModifierType.NEXT_TURN_ENERGY in character.modifier_map:
        modifier_data = character.modifier_map[ModifierType.NEXT_TURN_ENERGY]
        modifier_data.stacks_current = min(modifier_data.stacks_current + value, STACKS_MAX)

        return [], []

    character.modifier_map[ModifierType.NEXT_TURN_ENERGY] = ModifierData(
        IS_BUFF,
        True,
        min(value, STACKS_MAX),
        STACKS_MIN,
        STACKS_MAX,
        STACKS_DURATION,
    )

    return [], []
