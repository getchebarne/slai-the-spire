from dataclasses import dataclass
from typing import Optional

from src.game.combat.state import Actor
from src.game.combat.state import ModifierType


ModifierViewType = ModifierType


@dataclass
class HealthView:
    current: int
    max: int


@dataclass
class BlockView:
    current: int


@dataclass
class ModifierView:
    type: ModifierViewType
    stacks: Optional[int]

    def __hash__(self) -> int:
        return hash(id(self))


@dataclass
class ActorView:
    name: str
    health: HealthView
    block: BlockView
    modifiers: set[ModifierView]


def _modifier_views(actor: Actor) -> set[ModifierView]:
    return {
        ModifierView(modifier_type, modifier.stacks)
        for modifier_type, modifier in actor.modifiers.items()
    }


def _actor_to_view(actor: Actor) -> ActorView:
    modifiers = _modifier_views(actor)

    return ActorView(
        actor.name,
        HealthView(actor.health.current, actor.health.max),
        BlockView(actor.block.current),
        modifiers,
    )
