import random
from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierConfig
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.entity.monster import MonsterMove
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


_NAME = "Louse (green)"
_HEALTH_MAX_MIN = 10
_HEALTH_MAX_MIN_ASC_7 = 11
_HEALTH_MAX_MAX = 15
_HEALTH_MAX_MAX_ASC_7 = 16
_BITE_DAMAGE_MIN = 5
_BITE_DAMAGE_MAX = 7
_GROW_STRENGTH = 3
_GROW_STRENGTH_ASC_17 = 4
_CURL_UP_CONFIG = ModifierConfig(is_buff=True, stacks_duration=False)
_CURL_UP_STACKS_CURRENT_MIN = 3
_CURL_UP_STACKS_CURRENT_MAX = 7
_CURL_UP_STACKS_CURRENT_MIN_ASC_7 = 4
_CURL_UP_STACKS_CURRENT_MAX_ASC_7 = 8
_CURL_UP_STACKS_CURRENT_MIN_ASC_17 = 9
_CURL_UP_STACKS_CURRENT_MAX_ASC_17 = 12


@register_factory(_NAME)
def create_monster_louse_red(
    ascension_level: AscensionLevel,
) -> tuple[EntityMonster, Callable[[EntityMonster], str]]:
    if ascension_level < 7:
        health_max = random.randint(_HEALTH_MAX_MIN, _HEALTH_MAX_MAX)
    else:
        health_max = random.randint(_HEALTH_MAX_MIN_ASC_7, _HEALTH_MAX_MAX_ASC_7)

    health_current = health_max

    if ascension_level < 7:
        curl_up_stacks_current = random.randint(
            _CURL_UP_STACKS_CURRENT_MIN, _CURL_UP_STACKS_CURRENT_MAX
        )
    elif ascension_level < 17:
        curl_up_stacks_current = random.randint(
            _CURL_UP_STACKS_CURRENT_MIN_ASC_7, _CURL_UP_STACKS_CURRENT_MAX_ASC_7
        )
    else:
        curl_up_stacks_current = random.randint(
            _CURL_UP_STACKS_CURRENT_MIN_ASC_17, _CURL_UP_STACKS_CURRENT_MAX_ASC_17
        )

    return (
        EntityMonster(
            _NAME,
            health_current,
            health_max,
            moves={
                "Bite": _get_move_bite(ascension_level),
                "Grow": _get_move_grow(ascension_level),
            },
            modifier_map={
                ModifierType.CURL_UP: ModifierData(
                    config=_CURL_UP_CONFIG,
                    is_new=False,
                    stacks_current=curl_up_stacks_current,
                )
            },
        ),
        _get_move_name_fungi_beast,
    )


def _get_move_name_fungi_beast(monster: EntityMonster) -> str:
    num = random.randint(0, 98)  # TODO: abstract

    if num < 60:
        if monster.move_name_history[-2:] == ["Bite", "Bite"]:
            return "Grow"

        return "Bite"

    if monster.move_name_history and monster.move_name_history[-1] == "Grow":
        return "Bite"

    return "Grow"


def _get_move_bite(ascension_level: AscensionLevel) -> MonsterMove:
    damage = random.randint(_BITE_DAMAGE_MIN, _BITE_DAMAGE_MAX)
    if ascension_level >= 2:
        damage += 1

    return MonsterMove(
        [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, damage, EffectTargetType.CHARACTER)],
        Intent(damage=damage),
    )


def _get_move_grow(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 17:
        return MonsterMove(
            [
                Effect(EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH, EffectTargetType.SOURCE),
            ],
            Intent(buff=True),
        )

    return MonsterMove(
        [
            Effect(
                EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH_ASC_17, EffectTargetType.SOURCE
            ),
        ],
        Intent(buff=True),
    )
