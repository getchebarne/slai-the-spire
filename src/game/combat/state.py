from dataclasses import dataclass
from enum import Enum

from src.game.combat.effect_queue import EffectQueue
from src.game.entity.manager import EntityManager


class FSMState(Enum):
    DEFAULT = "DEFAULT"
    AWAIT_TARGET_CARD = "AWAIT_TARGET_CARD"
    AWAIT_TARGET_EFFECT = "AWAIT_TARGET_EFFECT"


@dataclass
class CombatState:
    entity_manager: EntityManager
    effect_queue: EffectQueue
    fsm_state: FSMState | None
