from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.monster import EntityMonster


ATTACK_DAMAGE = 12
DEFEND_BLOCK = 12


def _get_effects_attack() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, ATTACK_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_defend() -> list[Effect]:
    return [Effect(EffectType.BLOCK_GAIN, DEFEND_BLOCK, EffectTargetType.SOURCE)]


def create_monster_dummy(
    health_current: int, health_max: int, move_name_current: str | None
) -> EntityMonster:
    return EntityMonster(
        "Dummy",
        health_current,
        health_max,
        move_map={
            "Attack": _get_effects_attack(),
            "Defend": _get_effects_defend(),
        },
        move_name_current=move_name_current,
    )
