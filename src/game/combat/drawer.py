import os

from src.game.combat.view import CombatView
from src.game.combat.view import Energy
from src.game.combat.view import Character
from src.game.combat.view import Card
from src.game.combat.view import Monster
from src.game.combat.view import Health
from src.game.combat.view import Block


N_TERM_COLS, _ = os.get_terminal_size()


def _energy_str(energy: Energy) -> str:
    return f"{energy.current} / {energy.max}"


def _card_str(card: Card) -> str:
    return f"({card.cost}) {card.name}"


def _hand_str(hand: list[Card]) -> str:
    # TODO: If there's an active card, return its name in green
    str_ = " / ".join([_card_str(card_name) for i, card_name in enumerate(hand)])
    str_ = f"[ {str_} ]"
    return str_


def _health_str(health: Health) -> str:
    return f"{health.current} / {health.max}"


def _block_str(block: Block) -> str:
    return f"{block.current} / {block.max}"


def _creature_str(creature: Character | Monster) -> str:
    return f"{creature.name} {_health_str(creature.health)} {_block_str(creature.block)}"


def draw_view(view: CombatView) -> None:
    # Energy
    print(_energy_str(view.energy))

    # Monsters
    for monster in view.monsters:
        # Print to the right side of the terminal
        print(f"{_creature_str(monster):>{N_TERM_COLS}}")

    # Character
    print(_creature_str(view.character))
    print("\n")

    # Hand
    print(_hand_str(view.hand))

    # Separator
    print("-" * N_TERM_COLS)
