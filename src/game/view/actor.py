from dataclasses import dataclass

from src.game.entity.actor import EntityActor
from src.game.entity.actor import ModifierType


ViewModifierType = ModifierType


@dataclass(frozen=True)
class ViewActor:
    name: str
    health_current: int
    health_max: int
    block_current: int

    # Dictionary mapping modifier types to current stacks
    modifiers: dict[ViewModifierType, int | None]


def get_view_modifiers(actor: EntityActor) -> dict[ViewModifierType, int | None]:
    return {
        modifier_type: modifier.stacks_current
        for modifier_type, modifier in actor.modifier_map.items()
    }
