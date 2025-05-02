from dataclasses import replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_card_play(
    entity_manager: EntityManager, **kwargs
) -> tuple[list[Effect], list[Effect]]:
    id_target = kwargs["id_target"]

    target = entity_manager.entities[id_target]

    # Energy loss
    effects_top = [Effect(EffectType.ENERGY_LOSS, value=target.cost)]

    # Exhaust vs. discard
    if target.exhaust:
        effects_top.append(Effect(EffectType.CARD_EXHAUST, id_target=id_target))
    else:
        effects_top.append(Effect(EffectType.CARD_DISCARD, id_target=id_target))

    # Card's effects
    effects_top.extend([replace(effect, id_source=id_target) for effect in target.effects])

    return [], effects_top
