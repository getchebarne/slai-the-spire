import random

from src.game.entity.monster import EntityMonster


def get_move_name_fungi_beast(monster: EntityMonster) -> str:
    num = random.randint(0, 98)  # TODO: abstract

    if num < 60:
        if monster.move_name_history[-2:] == ["Bite", "Bite"]:
            return "Grow"

        return "Bite"

    if monster.move_name_history and monster.move_name_history[-1] == "Grow":
        return "Bite"

    return "Grow"
