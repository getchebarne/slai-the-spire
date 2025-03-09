from dataclasses import dataclass

from src.game.combat.state import CombatState
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


# TODO: maybe move this elsewhere
@dataclass
class CombatView:
    character: CharacterView
    monsters: list[MonsterView]
    hand: list[CardView]
    energy: EnergyView
    effect: EffectView
    entity_selectable_ids: list[int]  # TODO: revisit
    draw_pile: set[CardView]
    disc_pile: set[CardView]


def view_combat(combat_state: CombatState) -> CombatView:
    return CombatView(
        view_character(combat_state.entities),
        view_monsters(combat_state.entities),
        view_hand(combat_state.entities),
        view_energy(combat_state.entities),
        view_effect(combat_state.effect_queue),
        combat_state.entities.entity_selectable_ids.copy(),  # revisit
        view_draw_pile(combat_state.entities),
        view_discard_pile(combat_state.entities),
    )
