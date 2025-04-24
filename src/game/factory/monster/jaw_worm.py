import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster
from src.game.factory.lib import register_factory
from src.game.types import AscensionLevel


_NAME = "Jaw Worm"
_HEALTH_MAX_MIN = 40
_HEALTH_MAX_MAX = 44
_HEALTH_MAX_MIN_ASC_7 = 42
_HEALTH_MAX_MAX_ASC_7 = 46
_CHOMP_DAMAGE = 11
_CHOMP_DAMAGE_ASC_2 = 12
_BELLOW_STRENGTH = 3
_BELLOW_STRENGTH_ASC_2 = 4
_BELLOW_STRENGTH_ASC_17 = 5
_BELLOW_BLOCK = 6
_BELLOW_BLOCK_ASC_17 = 9
_THRASH_DAMAGE = 7
_THRASH_BLOCK = 5


@register_factory(_NAME)
def create_monster_jaw_worm(ascension_level: AscensionLevel) -> EntityMonster:
    if ascension_level < 7:
        health_max = random.randint(_HEALTH_MAX_MIN, _HEALTH_MAX_MAX)
    else:
        health_max = random.randint(_HEALTH_MAX_MIN_ASC_7, _HEALTH_MAX_MAX_ASC_7)

    health_current = health_max

    return EntityMonster(
        _NAME,
        health_current,
        health_max,
        move_map={
            "Chomp": _get_effects_chomp(ascension_level),
            "Bellow": _get_effects_bellow(ascension_level),
            "Thrash": _get_effects_thrash(),
        },
    )


def _get_effects_chomp(ascension_level: AscensionLevel) -> list[Effect]:
    if ascension_level < 2:
        return [Effect(EffectType.DAMAGE_DEAL, _CHOMP_DAMAGE, EffectTargetType.CHARACTER)]

    return [Effect(EffectType.DAMAGE_DEAL, _CHOMP_DAMAGE_ASC_2, EffectTargetType.CHARACTER)]


def _get_effects_bellow(ascension_level: AscensionLevel) -> list[Effect]:
    if ascension_level < 2:
        return [
            Effect(EffectType.MODIFIER_STRENGTH_GAIN, _BELLOW_STRENGTH, EffectTargetType.SOURCE),
            Effect(EffectType.BLOCK_GAIN, _BELLOW_BLOCK, EffectTargetType.SOURCE),
        ]

    if ascension_level < 17:
        return [
            Effect(
                EffectType.MODIFIER_STRENGTH_GAIN, _BELLOW_STRENGTH_ASC_2, EffectTargetType.SOURCE
            ),
            Effect(EffectType.BLOCK_GAIN, _BELLOW_BLOCK, EffectTargetType.SOURCE),
        ]

    return [
        Effect(
            EffectType.MODIFIER_STRENGTH_GAIN, _BELLOW_STRENGTH_ASC_17, EffectTargetType.SOURCE
        ),
        Effect(EffectType.BLOCK_GAIN, _BELLOW_BLOCK_ASC_17, EffectTargetType.SOURCE),
    ]


def _get_effects_thrash() -> list[Effect]:
    return [
        Effect(EffectType.DAMAGE_DEAL, _THRASH_DAMAGE, EffectTargetType.CHARACTER),
        Effect(EffectType.BLOCK_GAIN, _THRASH_BLOCK, EffectTargetType.SOURCE),
    ]
