from dataclasses import dataclass

from src.game.core.fsm import FSM
from src.game.entity.manager import EntityManager
from src.game.types_ import AscensionLevel
from src.game.types_ import EffectQueue


@dataclass
class GameState:
    ascension_level: AscensionLevel
    entity_manager: EntityManager
    effect_queue: EffectQueue
    fsm: FSM | None
