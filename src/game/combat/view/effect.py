from dataclasses import dataclass
from typing import Optional

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.entities import EffectType
from src.game.combat.state import State


EffectViewType = EffectType


@dataclass
class EffectView:
    type: EffectViewType
    # number_of_targets: Optional[int] = None  # TODO: unused for now


def view_effect(effect_queue: EffectQueue, state: State) -> Optional[EffectView]:
    if state != State.AWAIT_EFFECT_TARGET:
        return None

    return EffectView(effect_queue.next_effect_type())
