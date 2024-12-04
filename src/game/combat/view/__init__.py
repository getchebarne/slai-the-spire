from dataclasses import dataclass

from src.game.combat.manager import CombatManager
from src.game.combat.view.card import CardView
from src.game.combat.view.card import view_discard_pile
from src.game.combat.view.card import view_draw_pile
from src.game.combat.view.card import view_hand
from src.game.combat.view.character import CharacterView
from src.game.combat.view.character import view_character
from src.game.combat.view.effect import EffectView
from src.game.combat.view.effect import view_effect
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.energy import view_energy
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.monster import view_monsters
from src.game.combat.view.state import StateView
from src.game.combat.view.state import view_state


# TODO: maybe move this elsewhere
@dataclass
class CombatView:
    state: StateView
    character: CharacterView
    monsters: list[MonsterView]
    hand: list[CardView]
    energy: EnergyView
    effect: EffectView
    entity_selectable_ids: list[int]  # TODO: revisit
    draw_pile: set[CardView]
    disc_pile: set[CardView]


def view_combat(combat_manager: CombatManager) -> CombatView:
    return CombatView(
        view_state(combat_manager.state),
        view_character(combat_manager.entities),
        view_monsters(combat_manager.entities),
        view_hand(combat_manager.entities),
        view_energy(combat_manager.entities),
        view_effect(combat_manager.effect_queue, combat_manager.state),
        combat_manager.entities.entity_selectable_ids.copy(),  # revisit
        view_draw_pile(combat_manager.entities),
        view_discard_pile(combat_manager.entities),
    )
