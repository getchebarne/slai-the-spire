import random
from typing import Callable

from src.game.combat.entities import Effect
from src.game.combat.entities import EffectTargetType
from src.game.combat.entities import EffectType
from src.game.combat.entities import Monster
from src.game.combat.entities import MonsterMove


ais: dict[str, Callable] = {}


def register_ai(monster_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        ais[monster_name] = func
        return func

    return decorator


# TODO: moves should go in factories
def move_attack() -> MonsterMove:
    return MonsterMove("Attack", [Effect(EffectType.DEAL_DAMAGE, 10, EffectTargetType.CHARACTER)])


def move_defend() -> MonsterMove:
    return MonsterMove("Defend", [Effect(EffectType.GAIN_BLOCK, 10, EffectTargetType.SOURCE)])


@register_ai("Dummy")
def ai_dummy(monster: Monster) -> None:
    if monster.move is None:
        monster.move = random.choice([move_attack, move_defend])()

        return

    if monster.move.name == "Attack":
        monster.move = move_defend()

        return

    if monster.move.name == "Defend":
        monster.move = move_attack()

        return

    raise ValueError(f"Unsupported move name: {monster.move.name}")


def move_chomp() -> MonsterMove:
    return MonsterMove("Chomp", [Effect(EffectType.DEAL_DAMAGE, 12, EffectTargetType.CHARACTER)])


def move_thrash() -> MonsterMove:
    return MonsterMove(
        "Thrash",
        [
            Effect(EffectType.DEAL_DAMAGE, 7, EffectTargetType.CHARACTER),
            Effect(EffectType.GAIN_BLOCK, 5, EffectTargetType.SOURCE),
        ],
    )


def move_bellow() -> MonsterMove:
    return MonsterMove(
        "Bellow",
        [
            Effect(EffectType.GAIN_STR, 5, EffectTargetType.SOURCE),
            Effect(EffectType.GAIN_BLOCK, 9, EffectTargetType.SOURCE),
        ],
    )


@register_ai("Jaw Worm")
def ai_jaw_worm(monster: Monster) -> None:
    if monster.move is None:
        monster.move = move_chomp()

    else:
        num = random.randint(0, 99)
        if num < 25:
            if monster.move_history[-1].name == "Chomp":
                if random.random() < 0.5625:  # 56.25% chance
                    monster.move = move_bellow()
                else:
                    monster.move = move_thrash()
            else:
                monster.move = move_chomp()

        elif num < 55:
            # TODO: must be false if there's not at least 2 moves in history
            if len(monster.move_history) >= 2 and all(
                [move.name == "Thrash" for move in monster.move_history[-2:]]
            ):
                if random.random() < 0.357:  # 35.7% chance
                    monster.move = move_chomp()
                else:
                    monster.move = move_bellow()
            else:
                monster.move = move_thrash()

        else:
            if monster.move_history[-1].name == "Bellow":
                if random.random() < 0.416:  # 41.6% chance
                    monster.move = move_chomp()
                else:
                    monster.move = move_thrash()
            else:
                monster.move = move_bellow()

    # Append move
    monster.move_history.append(monster.move)  # TODO: store name only
