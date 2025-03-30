from dataclasses import dataclass

from src.game.combat.entities import EntityManager
from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import actor_to_view


@dataclass
class CharacterView(ActorView):
    pass


def view_character(entity_manager: EntityManager) -> CharacterView:
    actor_view = actor_to_view(entity_manager.entities[entity_manager.id_character])

    return CharacterView(
        actor_view.name,
        actor_view.health_current,
        actor_view.health_max,
        actor_view.block_current,
        actor_view.modifiers,
    )
