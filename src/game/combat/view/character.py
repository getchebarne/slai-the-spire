from dataclasses import dataclass

from src.game.combat.entities import Entities
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class CharacterView(ActorView):
    pass


def view_character(entities: Entities) -> CharacterView:
    actor_view = _actor_to_view(entities.get_character())

    return CharacterView(
        actor_view.name, actor_view.health, actor_view.block, actor_view.modifiers
    )
