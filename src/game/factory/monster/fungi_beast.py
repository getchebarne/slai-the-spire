import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster
from src.game.factory.lib import register_factory
from src.game.types import AscensionLevel


_NAME = "Fungi Beast"
_HEALTH_MAX_MIN = 22
_HEALTH_MAX_MIN_ASC_7 = 24
_HEALTH_MAX_MAX = 28
_BITE_DAMAGE = 6
_GROW_STRENGTH = 3
_GROW_STRENGTH_ASC_2 = 4
_GROW_STRENGTH_ASC_17 = 5
_SPORE_CLOUD_STACKS_CURRENT = 2
_SPORE_CLOUD_STACKS_MIN = 1
_SPORE_CLOUD_STACKS_MAX = 999
_SPORE_CLOUD_STACKS_DURATION = False


@register_factory(_NAME)
def create_monster_fungi_beast(ascension_level: AscensionLevel) -> EntityMonster:
    if ascension_level < 7:
        health_max = random.randint(_HEALTH_MAX_MIN, _HEALTH_MAX_MAX)
    else:
        health_max = random.randint(_HEALTH_MAX_MIN_ASC_7, _HEALTH_MAX_MAX)

    health_current = health_max

    return EntityMonster(
        _NAME,
        health_current,
        health_max,
        move_map={
            "Bite": _get_effects_bite(),
            "Grow": _get_effects_grow(ascension_level),
        },
        modifier_map={
            ModifierType.SPORE_CLOUD: ModifierData(
                _SPORE_CLOUD_STACKS_CURRENT,
                _SPORE_CLOUD_STACKS_MIN,
                _SPORE_CLOUD_STACKS_MAX,
                _SPORE_CLOUD_STACKS_DURATION,
            )
        },
    )


def _get_effects_bite() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, _BITE_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_grow(ascension_level: AscensionLevel) -> list[Effect]:
    if ascension_level < 2:
        return [
            Effect(EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH, EffectTargetType.SOURCE),
        ]

    if ascension_level < 17:
        return [
            Effect(
                EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH_ASC_2, EffectTargetType.SOURCE
            ),
        ]

    return [
        Effect(EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH_ASC_17, EffectTargetType.SOURCE),
    ]
