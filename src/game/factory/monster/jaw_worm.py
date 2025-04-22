import random

from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.monster import EntityMonster


HEALTH_MAX_MIN = 42
HEALTH_MAX_MAX = 42
CHOMP_DAMAGE = 12
BELLOW_STRENGTH = 5
BELLOW_BLOCK = 9
THRASH_DAMAGE = 7
THRASH_BLOCK = 5


def _get_effects_chomp() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, CHOMP_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_bellow() -> list[Effect]:
    return [
        Effect(EffectType.MODIFIER_STRENGTH_GAIN, BELLOW_STRENGTH, EffectTargetType.SOURCE),
        Effect(EffectType.BLOCK_GAIN, BELLOW_BLOCK, EffectTargetType.SOURCE),
    ]


def _get_effects_thrash() -> list[Effect]:
    return [
        Effect(EffectType.DAMAGE_DEAL, THRASH_DAMAGE, EffectTargetType.CHARACTER),
        Effect(EffectType.BLOCK_GAIN, THRASH_BLOCK, EffectTargetType.SOURCE),
    ]


def create_monster_jaw_worm() -> EntityMonster:
    # TODO: add ascension parameter
    health_max = random.randint(HEALTH_MAX_MIN, HEALTH_MAX_MAX)
    health_current = health_max

    return EntityMonster(
        "Jaw Worm",
        health_current,
        health_max,
        move_map={
            "Chomp": _get_effects_chomp(),
            "Bellow": _get_effects_bellow(),
            "Thrash": _get_effects_thrash(),
        },
    )
