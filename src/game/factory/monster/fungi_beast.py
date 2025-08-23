import random
from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.entity.monster import MonsterMove
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


_NAME = "Fungi Beast"
_HEALTH_MAX_MIN = 22
_HEALTH_MAX_MIN_ASC_7 = 24
_HEALTH_MAX_MAX = 28
_BITE_DAMAGE = 6
_GROW_STRENGTH = 3
_GROW_STRENGTH_ASC_2 = 4
_GROW_STRENGTH_ASC_17 = 5
_SPORE_CLOUD_IS_BUFF = True
_SPORE_CLOUD_STACKS_CURRENT = 2
_SPORE_CLOUD_STACKS_MIN = 1
_SPORE_CLOUD_STACKS_MAX = 999
_SPORE_CLOUD_STACKS_DURATION = False


@register_factory(_NAME)
def create_monster_fungi_beast(
    ascension_level: AscensionLevel,
) -> tuple[EntityMonster, Callable[[EntityMonster], str]]:
    if ascension_level < 7:
        health_max = random.randint(_HEALTH_MAX_MIN, _HEALTH_MAX_MAX)
    else:
        health_max = random.randint(_HEALTH_MAX_MIN_ASC_7, _HEALTH_MAX_MAX)

    health_current = health_max

    return (
        EntityMonster(
            _NAME,
            health_current,
            health_max,
            moves={
                "Bite": _get_move_bite(),
                "Grow": _get_move_grow(ascension_level),
            },
            modifier_map={
                ModifierType.SPORE_CLOUD: ModifierData(
                    _SPORE_CLOUD_IS_BUFF,
                    False,
                    _SPORE_CLOUD_STACKS_CURRENT,
                    _SPORE_CLOUD_STACKS_MIN,
                    _SPORE_CLOUD_STACKS_MAX,
                    _SPORE_CLOUD_STACKS_DURATION,
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


def _get_move_bite() -> MonsterMove:
    return MonsterMove(
        [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _BITE_DAMAGE, EffectTargetType.CHARACTER)],
        Intent(damage=_BITE_DAMAGE),
    )


def _get_move_grow(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 2:
        return MonsterMove(
            [
                Effect(EffectType.MODIFIER_STRENGTH_GAIN, _GROW_STRENGTH, EffectTargetType.SOURCE),
            ],
            Intent(buff=True),
        )

    if ascension_level < 17:
        return MonsterMove(
            [
                Effect(EffectType).MODIFIER_STRENGTH_GAIN,
                _GROW_STRENGTH_ASC_2,
                EffectTargetType.SOURCE,
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
