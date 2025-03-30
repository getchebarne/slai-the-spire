import random
from typing import Callable

from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.combat.entities import MonsterMove


ais: dict[str, Callable] = {}


def register_ai(monster_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        ais[monster_name] = func
        return func

    return decorator


@register_ai("Dummy")
def ai_dummy(move_current: MonsterMove | None, move_history: list[MonsterMove]) -> MonsterMove:
    move_attack = MonsterMove(
        "Attack", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
    )
    move_defend = MonsterMove(
        "Defend", [Effect(EffectType.GAIN_BLOCK, 12, EffectTargetType.SOURCE)]
    )
    if move_current is None:
        return random.choice([move_attack, move_defend])

    if move_current.name == "Attack":
        return move_defend

    if move_current.name == "Defend":
        return move_attack

    raise ValueError(f"Unsupported move name: {move_current.name}")


# TODO: update w/ new Monster changes
@register_ai("Jaw Worm")
def ai_jaw_worm(move_current: MonsterMove | None, move_history: list[MonsterMove]) -> MonsterMove:
    move_chomp = MonsterMove(
        "Chomp", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)]
    )
    move_bellow = MonsterMove(
        "Bellow",
        [
            Effect(EffectType.GAIN_STRENGTH, 5, EffectTargetType.SOURCE),
            Effect(EffectType.GAIN_BLOCK, 9, EffectTargetType.SOURCE),
        ],
    )
    move_thrash = MonsterMove(
        "Thrash",
        [
            Effect(EffectType.DEAL_DAMAGE, 7, EffectTargetType.CHARACTER),
            Effect(EffectType.GAIN_BLOCK, 5, EffectTargetType.SOURCE),
        ],
    )
    if move_current is None:
        return move_chomp

    num = random.randint(0, 99)
    if num < 25:
        if move_history[-1].name == "Chomp":
            if random.random() < 0.5625:  # 56.25% chance
                return move_bellow

            return move_thrash

        return move_chomp

    elif num < 55:
        # TODO: must be false if there's not at least 2 moves in history
        if len(move_history) >= 2 and all([move.name == "Thrash" for move in move_history[-2:]]):
            if random.random() < 0.357:  # 35.7% chance
                return move_chomp

            return move_bellow

        return move_thrash

    if move_history[-1].name == "Bellow":
        if random.random() < 0.416:  # 41.6% chance
            return move_chomp

        return move_thrash

    return move_bellow
