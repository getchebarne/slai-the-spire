from dataclasses import dataclass

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.view.card import CardView
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
    # draw pile
    # discard pile


def view_combat(entities: Entities, effect_queue: EffectQueue) -> CombatView:
    return CombatView(
        view_character(entities),
        view_monsters(entities),
        view_hand(entities),
        view_energy(entities),
        view_effect(effect_queue),
    )
