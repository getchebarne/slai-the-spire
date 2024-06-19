from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectNumberOfTargetsComponent
from src.game.ecs.manager import ECSManager


@dataclass
class EffectView:
    type: str  # TODO: make enum
    number_of_targets: Optional[int]


def get_effect_view(entity_id: int, manager: ECSManager) -> EffectView:
    type_ = _get_effect_type(entity_id, manager)
    effect_number_of_targets_component = manager.get_component_for_entity(
        entity_id, EffectNumberOfTargetsComponent
    )

    return EffectView(
        type_,
        (
            None
            if effect_number_of_targets_component is None
            else effect_number_of_targets_component.value
        ),
    )


def _get_effect_type(entity_id: int, manager: ECSManager) -> str:
    if manager.get_component_for_entity(entity_id, EffectDiscardCardComponent) is not None:
        return "Discard"

    # TODO: add other cases
