from src.game.core.effect import Effect
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


def process_effect_modifier_sharp_hide_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    target = kwargs["target"]

    del target.modifier_map[ModifierType.SHARP_HIDE]

    return [], []
