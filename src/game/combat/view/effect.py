from dataclasses import dataclass
from typing import Optional

from src.game.combat.effect_queue import EffectQueue
from src.game.combat.state import EffectType


EffectViewType = EffectType


@dataclass
class EffectView:
    type: EffectViewType
    # number_of_targets: Optional[int] = None  # TODO: unused for now


def view_effect(effect_queue: EffectQueue) -> Optional[EffectView]:
    if (source_id_effect_pending := effect_queue.get_pending()) is None:
        return None

    source_id_pending, effect_pending = source_id_effect_pending

    return EffectView(effect_pending.type)
