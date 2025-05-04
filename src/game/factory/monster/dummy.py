import random
from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


_NAME = "Dummy"
_HEALTH_MAX = 50
_ATTACK_DAMAGE = 12
_DEFEND_BLOCK = 12


@register_factory(_NAME)
def create_monster_dummy(
    ascension_level: AscensionLevel,
) -> tuple[EntityMonster, Callable[[EntityMonster], str]]:
    health_current = _HEALTH_MAX

    return EntityMonster(
        _NAME,
        health_current,
        _HEALTH_MAX,
        move_map={
            "Attack": _get_effects_attack(),
            "Defend": _get_effects_defend(),
        },
    )


def get_move_name_dummy(monster: EntityMonster) -> str:
    if monster.move_name_current_current is None:
        return random.choice(list(monster.move_map.keys()))

    if monster.move_name_current == "Attack":
        return "Defend"

    if monster.move_name_current == "Defend":
        return "Attack"

    raise ValueError(f"Unsupported move name: {monster.move_name_current}")


def _get_effects_attack() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _ATTACK_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_defend() -> list[Effect]:
    return [Effect(EffectType.BLOCK_GAIN, _DEFEND_BLOCK, EffectTargetType.SOURCE)]
