from dataclasses import dataclass
from typing import Optional

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.state import State


@dataclass
class CombatManager:
    entities: Entities
    effect_queue: EffectQueue
    state: Optional[State] = None
