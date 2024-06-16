import os
from typing import Optional

from src.game.combat.view import Block
from src.game.combat.view import Card
from src.game.combat.view import CombatView
from src.game.combat.view import Creature
from src.game.combat.view import EffectIsPendingInputTargets
from src.game.combat.view import Energy
from src.game.combat.view import Health
from src.game.combat.view import Intent


N_TERM_COLS, _ = os.get_terminal_size()


def _energy_str(energy: Energy) -> str:
    return f"\U0001F50B {energy.current}/{energy.max}"


def _card_str(card: Card) -> str:
    return f"({card.cost}) {card.name}"


def _hand_str(hand: list[Card]) -> str:
    # TODO: If there's an active card, return its name in green
    return " / ".join(
        [
            (f"\033[92m{_card_str(card)}\033[0m" if card.is_selected else _card_str(card))
            for card in hand
        ]
    )


def _health_str(health: Health) -> str:
    return f"\U0001FAC0 {health.current} / {health.max}"


def _block_str(block: Block) -> str:
    return f"\U0001F6E1 {block.current}"


def _creature_str(creature: Creature) -> str:
    return (
        f"{creature.name} {_health_str(creature.health)} {_block_str(creature.block)}"
        f"{creature.modifiers}"
    )


def _effect_is_pending_input_targets_str(
    effect_is_pending_input_targets: EffectIsPendingInputTargets,
) -> str:
    return (
        f"{effect_is_pending_input_targets.name} | "
        f"{effect_is_pending_input_targets.number_of_targets}"
    )


def _intent_str(intent: Optional[Intent]) -> str:
    str_ = ""
    if intent is None:
        return str_

    if intent.damage:
        str_ = f"{str_} \U0001F5E1 {intent.damage}"
        if intent.times > 1:
            str_ = f"{str_} x {intent.times}"

    if intent.block:
        str_ = f"{str_} \U0001F6E1"

    return str_


def _view_str(view: CombatView) -> str:
    # Effect
    effect_str = "None"
    if view.effect is not None:
        effect_str = _effect_is_pending_input_targets_str(view.effect)

    # Monsters
    monster_strs = [
        f"{_intent_str(monster.intent)} {_creature_str(monster)}" for monster in view.monsters
    ]
    right_justified_monsters = "\n".join(
        [f"{monster_str:>{N_TERM_COLS}}" for monster_str in monster_strs]
    )

    # Character
    character_str = _creature_str(view.character)

    # Energy & hand
    energy_hand_str = f"{_energy_str(view.energy)} | {_hand_str(view.hand)}"

    # Separator
    separator = "-" * N_TERM_COLS

    return (
        effect_str
        + "\n"
        + right_justified_monsters
        + "\n"
        + character_str
        + "\n\n"
        + energy_hand_str
        + "\n"
        + separator
    )


class CombatDrawer:
    def __init__(self, prev_view_str: Optional[str] = None):
        self.prev_view_str = prev_view_str

    def __call__(self, combat_view: CombatView) -> None:
        view_str = _view_str(combat_view)

        if view_str != self.prev_view_str:
            print(view_str)
            self.prev_view_str = view_str


drawer = CombatDrawer()
