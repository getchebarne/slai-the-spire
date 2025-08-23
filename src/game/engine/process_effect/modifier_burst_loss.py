from src.game.core.effect import Effect
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


# TODO: centralize
IS_BUFF = True
STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def process_effect_modifier_burst_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    modifier_data = target.modifier_map[ModifierType.BURST]
    modifier_data.stacks_current -= value
    if modifier_data.stacks_current < STACKS_MIN:
        del target.modifier_map[ModifierType.BURST]

    return [], []
