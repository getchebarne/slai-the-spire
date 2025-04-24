from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster
from src.game.factory.lib import register_factory
from src.game.types import AscensionLevel


_NAME = "Dummy"
_HEALTH_MAX = 50
_ATTACK_DAMAGE = 12
_DEFEND_BLOCK = 12


@register_factory(_NAME)
def create_monster_dummy(ascension_level: AscensionLevel) -> EntityMonster:
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


def _get_effects_attack() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, _ATTACK_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_defend() -> list[Effect]:
    return [Effect(EffectType.BLOCK_GAIN, _DEFEND_BLOCK, EffectTargetType.SOURCE)]
