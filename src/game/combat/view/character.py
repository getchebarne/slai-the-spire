from dataclasses import dataclass

from src.game.combat.entities import Entities
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import _actor_to_view


@dataclass
class CharacterView(ActorView):
    pass


def view_character(entities: Entities) -> CharacterView:
    actor_view = _actor_to_view(entities.all[entities.character_id])

    return CharacterView(
        actor_view.name,
        actor_view.health_current,
        actor_view.health_max,
        actor_view.block_current,
        # actor_view.modifiers,
    )
