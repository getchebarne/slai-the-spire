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


def move_attack() -> MonsterMove:
    return MonsterMove("Attack", [Effect(EffectType.DEAL_DAMAGE, 10, EffectTargetType.CHARACTER)])


def move_defend() -> MonsterMove:
    return MonsterMove("Defend", [Effect(EffectType.GAIN_BLOCK, 10, EffectTargetType.TURN)])


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
