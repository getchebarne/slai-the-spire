from dataclasses import dataclass

from src.game.combat.entities import Actor


# TODO: revisit
@dataclass
class ModifierView:
    is_stackable: bool
    stacks_current: int | None


@dataclass
class ActorView:
    name: str
    health_current: int
    health_max: int
    block_current: int
    modifier_weak: ModifierView


def actor_to_view(actor: Actor) -> ActorView:
    return ActorView(
        actor.name,
        actor.health_current,
        actor.health_max,
        actor.block_current,
        ModifierView(False, actor.modifier_weak.stacks_current),
    )
