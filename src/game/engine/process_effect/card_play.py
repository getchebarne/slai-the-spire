from dataclasses import replace

from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.entity.manager import EntityManager


def process_effect_card_play(
    entity_manager: EntityManager, effect: Effect
) -> tuple[list[Effect], list[Effect]]:
    id_target = effect.id_target
    target = entity_manager.entities[id_target]

    return (
        [],
        [
            # TODO; move to engine?
            Effect(EffectType.ENERGY_LOSS, value=target.cost),
            Effect(EffectType.CARD_DISCARD, id_target=id_target),
            *[replace(effect, id_source=id_target) for effect in target.effects],
        ],
    )
