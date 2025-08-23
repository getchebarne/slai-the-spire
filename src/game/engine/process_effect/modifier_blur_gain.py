from src.game.core.effect import Effect
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager


IS_BUFF = True
STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = True


def process_effect_modifier_blur_gain(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]
    id_source = kwargs["id_source"]

    target = entity_manager.entities[id_target]
    source = entity_manager.entities[id_source]

    if ModifierType.BLUR in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.BLUR]
        modifier_data.stacks_current = min(modifier_data.stacks_current + value, STACKS_MAX)

        return [], []

    if isinstance(source, EntityCard):
        created_by_character = True
    else:
        created_by_character = False

    target.modifier_map[ModifierType.BLUR] = ModifierData(
        IS_BUFF,
        created_by_character,
        min(value, STACKS_MAX),
        STACKS_MIN,
        STACKS_MAX,
        STACKS_DURATION,
    )

    return [], []
