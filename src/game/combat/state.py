from dataclasses import dataclass

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import EntityManager


@dataclass
class CombatState:
    entity_manager: EntityManager
    effect_queue: EffectQueue
