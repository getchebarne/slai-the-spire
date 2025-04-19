import random

from src.game.entity.monster import EntityMonster


def get_move_name_dummy(monster: EntityMonster) -> str:
    if monster.move_name_current_current is None:
        return random.choice(list(monster.move_map.keys()))

    if monster.move_name_current == "Attack":
        return "Defend"

    if monster.move_name_current == "Defend":
        return "Attack"

    raise ValueError(f"Unsupported move name: {monster.move_name_current}")
