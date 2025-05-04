import random
from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


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
def create_monster_jaw_worm(
    ascension_level: AscensionLevel,
) -> tuple[EntityMonster, Callable[[EntityMonster], str]]:
    if ascension_level < 7:
        health_max = random.randint(_HEALTH_MAX_MIN, _HEALTH_MAX_MAX)
    else:
        health_max = random.randint(_HEALTH_MAX_MIN_ASC_7, _HEALTH_MAX_MAX_ASC_7)

    health_current = health_max

    return (
        EntityMonster(
            _NAME,
            health_current,
            health_max,
            move_map={
                "Chomp": _get_effects_chomp(ascension_level),
                "Bellow": _get_effects_bellow(ascension_level),
                "Thrash": _get_effects_thrash(),
            },
        ),
        _get_move_name_jaw_worm,
    )


def _get_move_name_jaw_worm(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Chomp"

    num = random.randint(0, 98)  # TODO: abstract
    if num < 25:
        if monster.move_name_history[-1] == "Chomp":
            if random.random() < 0.5625:  # 56.25% chance
                return "Bellow"

            return "Thrash"

        return "Chomp"

    elif num < 55:
        if monster.move_name_history[-2:] == ["Thrash", "Thrash"]:
            if random.random() < 0.357:  # 35.7% chance
                return "Chomp"

            return "Bellow"

        return "Thrash"

    if monster.move_name_history[-1] == "Bellow":
        if random.random() < 0.416:  # 41.6% chance
            return "Chomp"

        return "Thrash"

    return "Bellow"


def _get_effects_chomp(ascension_level: AscensionLevel) -> list[Effect]:
    if ascension_level < 2:
        return [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _CHOMP_DAMAGE, EffectTargetType.CHARACTER)]

    return [
        Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _CHOMP_DAMAGE_ASC_2, EffectTargetType.CHARACTER)
    ]


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
        Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _THRASH_DAMAGE, EffectTargetType.CHARACTER),
        Effect(EffectType.BLOCK_GAIN, _THRASH_BLOCK, EffectTargetType.SOURCE),
    ]
