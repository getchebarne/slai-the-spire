from dataclasses import dataclass

from src.game.combat.view.card import CardView
from src.game.combat.view.card import get_hand_view
from src.game.combat.view.character import CharacterView
from src.game.combat.view.character import get_character_view
from src.game.combat.view.effect import EffectView
from src.game.combat.view.effect import get_effect_view
from src.game.combat.view.energy import EnergyView
from src.game.combat.view.energy import get_energy_view
from src.game.combat.view.monster import MonsterView
from src.game.combat.view.monster import get_monsters_view
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.manager import ECSManager


@dataclass
class CombatView:
    character: CharacterView
    monsters: list[MonsterView]
    hand: list[CardView]
    energy: EnergyView
    effect: EffectView


def get_combat_view(manager: ECSManager) -> CombatView:
    effect_entity_id = None
    query_result = list(manager.get_component(EffectIsPendingInputTargetsComponent))
    if query_result:
        effect_entity_id, _ = query_result[0]

    return CombatView(
        get_character_view(manager),
        get_monsters_view(manager),
        get_hand_view(manager),
        get_energy_view(manager),
        None if effect_entity_id is None else get_effect_view(effect_entity_id, manager),
    )
