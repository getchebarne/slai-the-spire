from dataclasses import dataclass
from typing import Optional

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import Entities
from src.game.combat.state import State
from src.game.combat.state import on_enter


@dataclass
class CombatManager:
    entities: Optional[Entities] = None
    effect_queue: Optional[EffectQueue] = None
    state: Optional[State] = None

    def __post_init__(self) -> None:
        if self.entities is None:
            self.entities = Entities()

        if self.effect_queue is None:
            self.effect_queue = EffectQueue()

        if self.state is None:
            on_enter(State.DEFAULT, self.entities)
            self.state = State.DEFAULT
