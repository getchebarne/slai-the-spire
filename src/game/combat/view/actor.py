from dataclasses import dataclass

from src.game.combat.entities import Actor


# TODO: reenable
# @dataclass
# class ModifierView:
#     type: ModifierViewType
#     stacks: Optional[int]

#     def __hash__(self) -> int:
#         return hash(id(self))


@dataclass
class ActorView:
    name: str
    health_current: int
    health_max: int
    block_current: int
    # modifiers: set[ModifierView]


# def _modifier_views(actor: Actor) -> set[ModifierView]:
#     return {
#         ModifierView(modifier_type, modifier.stacks)
#         for modifier_type, modifier in actor.modifiers.items()
#     }


def _actor_to_view(actor: Actor) -> ActorView:
    # modifiers = _modifier_views(actor)

    return ActorView(
        actor.name,
        actor.health_current,
        actor.health_max,
        actor.block_current,
        # modifiers,
    )
