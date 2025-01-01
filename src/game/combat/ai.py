import random
from typing import Callable


ais: dict[str, Callable] = {}


def register_ai(monster_name: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        ais[monster_name] = func
        return func

    return decorator


@register_ai("Dummy")
def ai_dummy(move_name_current: str | None, move_name_history: list[str]) -> str:
    if move_name_current is None:
        return random.choice(["Attack", "Defend"])

    if move_name_current == "Attack":
        return "Defend"

    if move_name_current == "Defend":
        return "Attack"

    raise ValueError(f"Unsupported move name: {move_name_current}")


@register_ai("Jaw Worm")
def ai_jaw_worm(move_name_current: str | None, move_name_history: list[str]) -> str:
    if move_name_current is None:
        return "Chomp"

    num = random.randint(0, 99)
    if num < 25:
        if move_name_history[-1] == "Chomp":
            if random.random() < 0.5625:  # 56.25% chance
                return "Bellow"

            return "Thrash"

        return "Chomp"

    elif num < 55:
        # TODO: must be false if there's not at least 2 moves in history
        if len(move_name_history) >= 2 and all(
            [move_name == "Thrash" for move_name in move_name_history[-2:]]
        ):
            if random.random() < 0.357:  # 35.7% chance
                return "Chomp"

            return "Bellow"

        return "Thrash"

    if move_name_history[-1] == "Bellow":
        if random.random() < 0.416:  # 41.6% chance
            return "Chomp"

        return "Thrash"

    return "Bellow"
