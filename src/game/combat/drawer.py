import os

from src.game.combat.view import ActorView
from src.game.combat.view import CardView
from src.game.combat.view import CombatView
from src.game.combat.view import EnergyView
from src.game.combat.view import IntentView
from src.game.combat.view import MonsterView


N_COL, _ = os.get_terminal_size()
WHITE = "\033[37;1m"
RED = "\033[31;1m"
CYAN = "\033[36;1m"
SELECTED = "\033[32;1m"
RESET = "\033[0m"


def _energy_str(energy: EnergyView) -> str:
    return f"ENERGY: {energy.current}/{energy.max}"


def _card_str(card: CardView) -> str:
    return f"({card.cost}) {card.name}"


def _hand_str(hand: list[CardView]) -> str:
    card_strings = []
    for card_view in hand:
        if card_view.is_active:
            card_strings.append(f"{SELECTED}{_card_str(card_view)}{RESET}")

            continue

        card_strings.append(f"{_card_str(card_view)}")

    return "HAND: " + " / ".join(card_strings)


def _health_str(health_current: int, health_max: int) -> str:
    return f"HP: {health_current}/{health_max}"


def _block_str(block_current: int) -> str:
    return f"BLK: {block_current}"


def _actor_str(actor: ActorView, n_col: int = 0) -> str:
    modifier_strs = "\n".join(
        [
            f"{modifier_view_type.name}: {stacks_current}"
            for modifier_view_type, stacks_current in actor.modifiers.items()
        ]
    )

    return (
        f"{WHITE}{actor.name:>{n_col}}{RESET}\n"
        f"{WHITE}{'-' * len(actor.name):>{n_col}}{RESET}\n"
        f"{RED}{_health_str(actor.health_current, actor.health_max):>{n_col}}{RESET}\n"
        f"{CYAN}{_block_str(actor.block_current):>{n_col}}{RESET}\n"
        f"{modifier_strs:>{n_col}}"
    )


# def _effect_str(effect_view: EffectView | None) -> str:
#     if effect_view is None:
#         return "None"

#     return f"{effect_view.type}: {effect_view.number_of_targets}"


def _intent_str(intent_view: IntentView | None) -> str:
    str_ = ""
    if intent_view is None:
        return str_

    if intent_view.damage is not None:
        str_ = f"{str_}{intent_view.damage} x {intent_view.instances}"

    if intent_view.block:
        if str_ != "":
            str_ = f"{str_} & Blocking"

        else:
            str_ = "Blocking"

    if intent_view.buff:
        if str_ != "":
            str_ = f"{str_} & Buffing"

        else:
            str_ = "Buffing"

    return str_


def _monster_str(monster_view: MonsterView) -> str:
    # Get base actor string
    str_ = _actor_str(monster_view, N_COL)

    # Split into lines
    lines = str_.split("\n")

    # Insert monster's intent at first position
    lines.insert(0, _intent_str(monster_view.intent))

    # Align lines to the right of the terminal
    right_aligned_lines = [f"{line:>{N_COL}}" for line in lines]

    # Stitch together and return
    return "\n".join(right_aligned_lines)


def draw_combat(combat_view: CombatView) -> str:
    # Effect
    # effect_str = _effect_str(view.effect)

    # Monsters
    monster_strs = "\n".join([f"{_monster_str(monster)}" for monster in combat_view.monsters])

    # Character
    character_str = _actor_str(combat_view.character)

    # Energy
    energy_str = _energy_str(combat_view.energy)

    # Hand
    hand_str = _hand_str(combat_view.hand)

    # Separator
    separator = "-" * N_COL

    _str = (
        monster_strs
        + "\n"
        + character_str
        + "\n"
        + energy_str
        + "\n"
        + hand_str
        + "\n"
        + separator
    )

    print(_str)
