from dataclasses import dataclass

from src.game.combat.context import GameContext
from src.game.combat.view.card import CardView
from src.game.combat.view.card import view_hand
from src.game.combat.view.character import CharacterView
from src.game.combat.view.character import view_character
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.energy import view_energy
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.monster import view_monsters


# TODO: maybe move this elsewhere
@dataclass
class CombatView:
    character: CharacterView
    monsters: list[MonsterView]
    hand: list[CardView]
    energy: EnergyView
    # effect: EffectView
    # draw pile
    # discard pile


def view_combat(context: GameContext) -> CombatView:
    return CombatView(
        view_character(context), view_monsters(context), view_hand(context), view_energy(context)
    )
