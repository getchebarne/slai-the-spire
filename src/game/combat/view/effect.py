from dataclasses import dataclass
from typing import Optional

from src.game.combat.state import EffectType
from src.game.combat.state import GameState


EffectViewType = EffectType


@dataclass
class EffectView:
    type: EffectViewType
    number_of_targets: Optional[int] = None  # TODO: unused for now


def view_effect(state: GameState) -> Optional[EffectView]:
    if state.effect_type is None:
        return None

    return EffectView(state.effect_type)
