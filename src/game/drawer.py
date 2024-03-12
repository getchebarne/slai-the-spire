from functools import wraps
from typing import Any, Callable

from game import context


def draw_state() -> None:
    print("State: ", context.state)
    print("Hand: ", context.hand)
    print("Draw pile: ", context.draw_pile)
    print("Discard pile: ", context.disc_pile)
    print("Energy: ", context.energy)
    print(
        "Active card: ",
        context.hand[context.active_card_idx] if context.active_card_idx is not None else None,
    )
    print("Character: ", context.char)
    print("Monsters: ", context.monsters)
    print("-" * 100)


def draw_state_dec(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        draw_state()
        return func(*args, **kwargs)

    return wrapper
