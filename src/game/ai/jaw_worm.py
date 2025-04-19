import random

from src.game.entity.monster import EntityMonster


def get_move_name_jaw_worm(monster: EntityMonster) -> str:
    if monster.move_name_current is None:
        return "Chomp"

    num = random.randint(0, 98)  # TODO: abstract
    if num < 25:
        if monster.move_name_history[-1] == "Chomp":
            if random.random() < 0.5625:  # 56.25% chance
                return "Bellow"

            return "Thrash"

        return "Chomp"

    elif num < 55:
        if monster.move_name_history[-2:] == ["Thrash", "Thrash"]:
            if random.random() < 0.357:  # 35.7% chance
                return "Chomp"

            return "Bellow"

        return "Thrash"

    if monster.move_name_history[-1] == "Bellow":
        if random.random() < 0.416:  # 41.6% chance
            return "Chomp"

        return "Thrash"

    return "Bellow"
