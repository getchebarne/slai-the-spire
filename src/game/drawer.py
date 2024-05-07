import os
from functools import wraps
from typing import Any, Callable

from src.game.core.state import BattleState
from src.game.context import Context
from src.game.context import EntityData
from src.game.core.energy import Energy
from src.game.lib.card import card_lib


N_TERM_COLS, _ = os.get_terminal_size()


def state_str(state: BattleState) -> str:
    if state == BattleState.DEFAULT:
        return "Default"
    if state == BattleState.AWAIT_TARGET:
        return "Awaiting target"


def energy_str(energy: Energy) -> str:
    return f"{energy.current}/{energy.max} \U0001F50B"


def card_str(card_name: str) -> str:
    return f"({card_lib[card_name].cost}) {card_name}"


def hand_str(hand: list[str], active_card_idx: int) -> str:
    # If there's an active card, return its name in green
    str_ = " / ".join(
        [
            (
                f"\033[92m{card_str(card_name)}\033[0m"
                if i == active_card_idx
                else card_str(card_name)
            )
            for i, card_name in enumerate(hand)
        ]
    )
    str_ = f"[ {str_} ]"
    return str_


def entity_str(entity: EntityData) -> str:
    return (
        f"{entity.name} "
        f"\u2764\uFE0F {entity.current_health}/{entity.max_health}"
        f"\U0001F6E1 {entity.current_block}"
    )


def draw(context: Context) -> None:
    print(state_str(context.state))
    print(energy_str(context.energy))
    print(hand_str(context.hand, context.active_card_idx))
    for _, monster_data in context.get_monster_data():
        # Print to the right side of the terminal
        print(f"{entity_str(monster_data):>{N_TERM_COLS}}")

    print(entity_str(context.entities[context.CHAR_ENTITY_ID]))
    print("-" * N_TERM_COLS)


def draw_state_dec(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        draw()
        return func(*args, **kwargs)

    return wrapper
