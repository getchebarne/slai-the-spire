from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.manager import EntityManager


def process_effect_health_loss(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    value = kwargs["value"]
    target = kwargs["target"]

    # Decrease health
    target.health_current = max(0, target.health_current - value)

    if target.health_current == 0:
        return [], [Effect(EffectType.DEATH, target=target)]

    effects_top = []
    if ModifierType.MODE_SHIFT in target.modifier_map:
        modifier_data = target.modifier_map[ModifierType.MODE_SHIFT]
        modifier_data.stacks_current -= value
        if modifier_data.stacks_current < modifier_data.config.stacks_min:
            del target.modifier_map[ModifierType.MODE_SHIFT]

            # TODO: should add block here
            effects_top.append(Effect(EffectType.MONSTER_MOVE_UPDATE, target=target))

    return [], effects_top
