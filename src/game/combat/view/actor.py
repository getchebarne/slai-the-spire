from dataclasses import dataclass

from src.game.combat.context import Actor


@dataclass
class HealthView:
    current: int
    max: int


@dataclass
class BlockView:
    current: int


@dataclass
class ActorView:
    name: str
    health: HealthView
    block: BlockView


def _actor_to_view(actor: Actor) -> ActorView:
    return ActorView(
        actor.name,
        HealthView(actor.health.current, actor.health.max),
        BlockView(actor.block.current),
    )
