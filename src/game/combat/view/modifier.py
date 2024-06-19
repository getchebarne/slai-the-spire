from dataclasses import dataclass
from typing import Optional

from src.game.ecs.components.actors import ModifierStacksComponent
from src.game.ecs.components.actors import ModifierWeakComponent
from src.game.ecs.manager import ECSManager


@dataclass
class ModifierView:
    type: str  # TODO: make enum
    stacks: Optional[int]


def get_modifier_view(entity_id: int, manager: ECSManager) -> ModifierView:
    type_ = _get_modifier_type(entity_id, manager)
    modifier_stacks_component = manager.get_component_for_entity(
        entity_id, ModifierStacksComponent
    )

    return ModifierView(
        type_, None if modifier_stacks_component is None else modifier_stacks_component.value
    )


def _get_modifier_type(entity_id: int, manager: ECSManager) -> str:
    if manager.get_component_for_entity(entity_id, ModifierWeakComponent) is not None:
        return "Weak"
