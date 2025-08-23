from typing import Callable

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster
from src.game.entity.monster import Intent
from src.game.entity.monster import MonsterMove
from src.game.entity.monster import MonsterType
from src.game.factory.lib import register_factory
from src.game.types_ import AscensionLevel


_NAME = "The Guardian"
_CHARGING_UP_BLOCK = 9
_DEFENSIVE_MODE_SHARP_HIDE = 3
_DEFENSIVE_MODE_SHARP_HIDE_ASC_19 = 4
_FIERCE_BASH_DAMAGE = 32
_FIERCE_BASH_DAMAGE_ASC_4 = 36
_HEALTH_MAX = 240
_HEALTH_MAX_ASC_9 = 250
_MODE_SHIFT_IS_BUFF = True
_MODE_SHIFT_STACKS_CURRENT = 30
_MODE_SHIFT_STACKS_CURRENT_ASC_9 = 35
_MODE_SHIFT_STACKS_CURRENT_ASC_19 = 40
_MODE_SHIFT_STACKS_MIN = 1
_MODE_SHIFT_STACKS_MAX = 999
_MODE_SHIFT_STACKS_DURATION = False
_ROLL_ATTACK_DAMAGE = 9
_ROLL_ATTACK_DAMAGE_ASC_4 = 10
_TWIN_SLAM_DAMAGE = 8
_TWIN_SLAM_INSTANCES = 2
_VENT_STEAM_WEAK = 2
_VENT_STEAM_VULN = 2
_WHIRLWIND_DAMAGE = 5
_WHIRLWIND_INSTANCES = 4


@register_factory(_NAME)
def create_monster_the_guardian(
    ascension_level: AscensionLevel,
) -> tuple[EntityMonster, Callable[[EntityMonster], str]]:
    if ascension_level < 9:
        health_max = _HEALTH_MAX
    else:
        health_max = _HEALTH_MAX_ASC_9

    health_current = health_max

    return (
        EntityMonster(
            _NAME,
            health_current,
            health_max,
            type=MonsterType.BOSS,
            moves={
                "Charging Up": _get_move_charging_up(),
                "Fierce Bash": _get_move_fierce_bash(ascension_level),
                "Vent Steam": _get_move_vent_steam(),
                "Whirlwind": _get_move_whirlwind(),
                "Defensive Mode": _get_move_defensive_mode(ascension_level),
                "Roll Attack": _get_move_roll_attack(ascension_level),
                "Twin Slam": _get_move_twin_slam(),
            },
            modifier_map=_get_modifier_map(ascension_level),
        ),
        _get_move_name_the_guardian,
    )


def _get_move_name_the_guardian(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Charging Up"

    if ModifierType.MODE_SHIFT in monster.modifier_map:
        # Offensive mode
        if monster.move_name_history[-1] == "Charging Up":
            return "Fierce Bash"

        if monster.move_name_history[-1] == "Fierce Bash":
            return "Vent Steam"

        if monster.move_name_history[-1] == "Vent Steam":
            return "Whirlwind"

        if monster.move_name_history[-1] == "Whirlwind":
            return "Charging Up"

        if monster.move_name_history[-1] == "Twin Slam":
            return "Whirlwind"

        raise ValueError(f"Unsupported move name: {monster.move_name_history[-1]}")

    # Defensive mode
    if ModifierType.SHARP_HIDE in monster.modifier_map:
        if monster.move_name_history[-1] == "Defensive Mode":
            return "Roll Attack"

        if monster.move_name_history[-1] == "Roll Attack":
            return "Twin Slam"

        raise ValueError(f"Unsupported move name: {monster.move_name_history[-1]}")

    return "Defensive Mode"


def _get_move_charging_up() -> MonsterMove:
    return MonsterMove(
        [Effect(EffectType.BLOCK_GAIN, _CHARGING_UP_BLOCK, EffectTargetType.SOURCE)],
        Intent(block=True),
    )


def _get_move_fierce_bash(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 4:
        return MonsterMove(
            [
                Effect(
                    EffectType.DAMAGE_DEAL_PHYSICAL,
                    _FIERCE_BASH_DAMAGE,
                    EffectTargetType.CHARACTER,
                ),
            ],
            Intent(damage=_FIERCE_BASH_DAMAGE),
        )

    return MonsterMove(
        [
            Effect(
                EffectType.DAMAGE_DEAL_PHYSICAL,
                _FIERCE_BASH_DAMAGE_ASC_4,
                EffectTargetType.CHARACTER,
            ),
        ],
        Intent(_FIERCE_BASH_DAMAGE_ASC_4),
    )


def _get_move_vent_steam() -> MonsterMove:
    return MonsterMove(
        [
            Effect(EffectType.MODIFIER_WEAK_GAIN, _VENT_STEAM_WEAK, EffectTargetType.CHARACTER),
            Effect(
                EffectType.MODIFIER_VULNERABLE_GAIN, _VENT_STEAM_VULN, EffectTargetType.CHARACTER
            ),
        ],
        Intent(debuff_powerful=True),
    )


def _get_move_whirlwind() -> MonsterMove:
    return MonsterMove(
        [
            Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _WHIRLWIND_DAMAGE, EffectTargetType.CHARACTER)
            for _ in range(_WHIRLWIND_INSTANCES)
        ],
        Intent(damage=_WHIRLWIND_DAMAGE, instances=_WHIRLWIND_INSTANCES),
    )


def _get_move_defensive_mode(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 19:
        return MonsterMove(
            [
                Effect(
                    EffectType.MODIFIER_SHARP_HIDE_GAIN,
                    _DEFENSIVE_MODE_SHARP_HIDE,
                    EffectTargetType.SOURCE,
                )
            ],
            Intent(buff=True),
        )

    return MonsterMove(
        [
            Effect(
                EffectType.MODIFIER_SHARP_HIDE_GAIN,
                _DEFENSIVE_MODE_SHARP_HIDE_ASC_19,
                EffectTargetType.SOURCE,
            )
        ],
        Intent(buff=True),
    )


def _get_move_roll_attack(ascension_level: AscensionLevel) -> MonsterMove:
    if ascension_level < 4:
        return MonsterMove(
            [
                Effect(
                    EffectType.DAMAGE_DEAL_PHYSICAL,
                    _ROLL_ATTACK_DAMAGE,
                    EffectTargetType.CHARACTER,
                )
            ],
            Intent(damage=_ROLL_ATTACK_DAMAGE),
        )

    return MonsterMove(
        [
            Effect(
                EffectType.DAMAGE_DEAL_PHYSICAL,
                _ROLL_ATTACK_DAMAGE_ASC_4,
                EffectTargetType.CHARACTER,
            )
        ],
        Intent(damage=_ROLL_ATTACK_DAMAGE_ASC_4),
    )


def _get_move_twin_slam() -> MonsterMove:
    return MonsterMove(
        [
            Effect(EffectType.DAMAGE_DEAL_PHYSICAL, _TWIN_SLAM_DAMAGE, EffectTargetType.CHARACTER)
            for _ in range(_TWIN_SLAM_INSTANCES)
        ]
        + [
            Effect(EffectType.MODIFIER_MODE_SHIFT_GAIN, target_type=EffectTargetType.SOURCE),
            Effect(EffectType.MODIFIER_SHARP_HIDE_LOSS, target_type=EffectTargetType.SOURCE),
        ],
        Intent(damage=_TWIN_SLAM_DAMAGE, instances=_TWIN_SLAM_INSTANCES, buff=True),
    )


def _get_modifier_map(ascension_level: AscensionLevel) -> dict[ModifierType, ModifierData]:
    if ascension_level < 9:
        stacks_current = _MODE_SHIFT_STACKS_CURRENT

    elif ascension_level < 19:
        stacks_current = _MODE_SHIFT_STACKS_CURRENT_ASC_9

    else:
        stacks_current = _MODE_SHIFT_STACKS_CURRENT_ASC_19

    return {
        ModifierType.MODE_SHIFT: ModifierData(
            _MODE_SHIFT_IS_BUFF,
            False,
            stacks_current,
            _MODE_SHIFT_STACKS_MIN,
            _MODE_SHIFT_STACKS_MAX,
            _MODE_SHIFT_STACKS_DURATION,
        )
    }
