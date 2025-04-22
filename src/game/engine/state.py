from dataclasses import dataclass

from src.game.core.combat_state import CombatState
from src.game.core.room import RoomType
from src.game.engine.effect_queue import EffectQueue
from src.game.entity.manager import EntityManager


@dataclass
class GameState:
    entity_manager: EntityManager
    effect_queue: EffectQueue
    room_type: RoomType | None = None
    combat_state: CombatState | None = None
