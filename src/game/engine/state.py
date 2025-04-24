from dataclasses import dataclass

from src.game.core.fsm import FSM
from src.game.engine.effect_queue import EffectQueue
from src.game.entity.manager import EntityManager
from src.game.map import Map
from src.game.types import AscensionLevel


@dataclass
class GameState:
    ascension_level: AscensionLevel
    entity_manager: EntityManager
    effect_queue: EffectQueue
    fsm: FSM | None
    map_: Map
