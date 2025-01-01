from dataclasses import dataclass
from typing import Optional

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.state import State
from src.game.combat.state import on_enter


@dataclass
class CombatManager:
    entities: Entities
    effect_queue: EffectQueue
    state: Optional[State] = None


def change_state(
    combat_manager: CombatManager, state: State, entity_selectable_ids: Optional[list[int]] = None
) -> None:
    combat_manager.state = state

    on_enter(combat_manager.entities, state, entity_selectable_ids)
