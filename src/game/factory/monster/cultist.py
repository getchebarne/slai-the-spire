import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster


HEALTH_MAX_MIN = 42
HEALTH_MAX_MAX = 42
DARK_STRIKE_DAMAGE = 6
INCANTATION_MODIFIER_RITUAL_GAIN = 5


def _get_effects_incantation() -> list[Effect]:
    return [
        Effect(
            EffectType.MODIFIER_RITUAL_GAIN,
            INCANTATION_MODIFIER_RITUAL_GAIN,
            EffectTargetType.SOURCE,
        ),
    ]


def _get_effects_dark_strike() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, DARK_STRIKE_DAMAGE, EffectTargetType.CHARACTER)]


def create_monster_cultist() -> EntityMonster:
    # TODO: add ascension parameter
    health_max = random.randint(HEALTH_MAX_MIN, HEALTH_MAX_MAX)
    health_current = health_max

    return EntityMonster(
        "Cultist",
        health_current,
        health_max,
        move_map={
            "Incantation": _get_effects_incantation(),
            "Dark Strike": _get_effects_dark_strike(),
        },
    )
