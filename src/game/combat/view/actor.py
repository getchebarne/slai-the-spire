from dataclasses import dataclass

from src.game.combat.entities import Actor
from src.game.combat.entities import ModifierType


# Alias of ModiferType
ModifierViewType = ModifierType


@dataclass
class ActorView:
    name: str
    health_current: int
    health_max: int
    block_current: int
    modifiers: dict[ModifierViewType, int | None]


def actor_to_view(actor: Actor) -> ActorView:
    return ActorView(
        actor.name,
        actor.health_current,
        actor.health_max,
        actor.block_current,
        {
            modifier_type: modifier.stacks_current
            for modifier_type, modifier in actor.modifiers.items()
        },
    )
