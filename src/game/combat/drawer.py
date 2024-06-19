import os
from typing import Optional

from src.game.combat.view import CombatView
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import BlockView
from src.game.combat.view.actor import HealthView
from src.game.combat.view.card import CardView
from src.game.combat.view.effect import EffectView
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.monster import IntentView
from src.game.combat.view.monster import MonsterView


N_TERM_COLS, _ = os.get_terminal_size()


def _energy_str(energy: EnergyView) -> str:
    return f"\U0001F50B {energy.current}/{energy.max}"


def _card_str(card: CardView) -> str:
    return f"({card.cost}) {card.name}"


def _hand_str(hand: list[CardView]) -> str:
    return " / ".join(
        [
            (f"\033[92m{_card_str(card)}\033[0m" if card.is_selected else _card_str(card))
            for card in hand
        ]
    )


def _health_str(health: HealthView) -> str:
    return f"\U0001FAC0 {health.current} / {health.max}"


def _block_str(block: BlockView) -> str:
    return f"\U0001F6E1 {block.current}"


def _actor_str(actor: ActorView) -> str:
    return (
        f"{actor.name} {_health_str(actor.health)} {_block_str(actor.block)}" f"{actor.modifiers}"
    )


def _effect_str(effect_view: Optional[EffectView]) -> str:
    if effect_view is None:
        return "None"

    return f"{effect_view.type} | {effect_view.number_of_targets}"


def _intent_str(intent_view: Optional[IntentView]) -> str:
    str_ = ""
    if intent_view is None:
        return str_

    if intent_view.damage is not None:
        str_ = f"{str_} \U0001F5E1 {intent_view.damage}"

    if intent_view.times is not None:
        str_ = f"{str_} x {intent_view.times}"

    if intent_view.block:
        str_ = f"{str_} \U0001F6E1"

    return str_


def _monster_str(monster_view: MonsterView) -> str:
    return f"{_intent_str(monster_view.intent)} | {_actor_str(monster_view)}"


def _view_str(view: CombatView) -> str:
    # Effect
    effect_str = _effect_str(view.effect)

    # Monsters
    monster_strs = [f"{_monster_str(monster)}" for monster in view.monsters]
    right_justified_monsters = "\n".join(
        [f"{monster_str:>{N_TERM_COLS}}" for monster_str in monster_strs]
    )

    # Character
    character_str = _actor_str(view.character)

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
