from dataclasses import dataclass

from src.game.combat.entities import EffectType
from src.game.combat.state import QueuedEffect


EffectViewType = EffectType


@dataclass
class EffectView:
    type: EffectViewType
    # number_of_targets: Optional[int] = None  # TODO: unused for now


def view_effect(effect_queue: list[QueuedEffect]) -> EffectView | None:
    if effect_queue:
        return EffectView(effect_queue[0].effect.type)

    return None
