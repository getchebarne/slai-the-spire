import random
from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.entity.monster import MonsterMove
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


_NAME = "Cultist"
_HEALTH_MAX_MIN = 48
_HEALTH_MAX_MAX = 54
_HEALTH_MAX_MIN_ASC_7 = 50
_HEALTH_MAX_MAX_ASC_7 = 56
_DARK_STRIKE_DAMAGE = 6
_INCANTATION_MODIFIER_RITUAL_GAIN = 3
_INCANTATION_MODIFIER_RITUAL_GAIN_ASC_2 = 4
_INCANTATION_MODIFIER_RITUAL_GAIN_ASC_17 = 5


@register_factory(_NAME)
def create_monster_cultist(
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
            moves={
                "Incantation": _get_move_incantation(ascension_level),
                "Dark Strike": _get_move_dark_strike(),
            },
        ),
        _get_move_name_cultist,
    )


def _get_move_name_cultist(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Incantation"

    return "Dark Strike"


def _get_move_incantation(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 2:
        return MonsterMove(
            [
                Effect(
                    EffectType.MODIFIER_RITUAL_GAIN,
                    _INCANTATION_MODIFIER_RITUAL_GAIN,
                    EffectTargetType.SOURCE,
                ),
            ],
            Intent(buff=True),
        )

    if ascension_level < 17:
        return MonsterMove(
            [
                Effect(
                    EffectType.MODIFIER_RITUAL_GAIN,
                    _INCANTATION_MODIFIER_RITUAL_GAIN_ASC_2,
                    EffectTargetType.SOURCE,
                ),
            ],
            Intent(buff=True),
        )

    return MonsterMove(
        [
            Effect(
                EffectType.MODIFIER_RITUAL_GAIN,
                _INCANTATION_MODIFIER_RITUAL_GAIN_ASC_17,
                EffectTargetType.SOURCE,
            ),
        ],
        Intent(buff=True),
    )


def _get_move_dark_strike() -> MonsterMove:
    return MonsterMove(
        [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _DARK_STRIKE_DAMAGE, EffectTargetType.CHARACTER)],
        Intent(damage=_DARK_STRIKE_DAMAGE),
    )
