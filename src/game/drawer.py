import os
from functools import wraps
from typing import Any, Callable

from game import context
from game.core.entity import Block
from game.core.entity import Entity
from game.core.entity import Health
from game.lib.card import card_lib


N_TERM_COLS, _ = os.get_terminal_size()


def state_str() -> str:
    if context.state == context.BattleState.DEFAULT:
        return "Default"
    if context.state == context.BattleState.AWAIT_TARGET:
        return "Awaiting target"


def energy_str() -> str:
    return f"{context.energy.current}/{context.energy.max} \U0001F50B"


def card_str(card_name: str) -> str:
    return f"({card_lib[card_name].card_cost}) {card_name}"


def hand_str() -> str:
    # If there's an active card, return its name in green
    str_ = " / ".join(
        [
            (
                f"\033[92m{card_str(card_name)}\033[0m"
                if i == context.active_card_idx
                else card_str(card_name)
            )
            for i, card_name in enumerate(context.hand)
        ]
    )
    str_ = f"[ {str_} ]"
    return str_


def health_str(health: Health) -> str:
    return f"\u2764\uFE0F {health.current}/{health.max}"


def block_str(block: Block) -> str:
    return f"\U0001F6E1 {block.current}"


def entity_str(entity: Entity) -> str:
    return f"{entity.name} {health_str(entity.health)} {block_str(entity.block)}"


def draw() -> None:
    print(state_str())
    print(energy_str())
    print(hand_str())
    for monster in context.monsters:
        # Print to the right side of the terminal
        print(f"{entity_str(monster):>{N_TERM_COLS}}")

    print(entity_str(context.char))
    print("-" * N_TERM_COLS)


def draw_state_dec(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        draw()
        return func(*args, **kwargs)

    return wrapper
