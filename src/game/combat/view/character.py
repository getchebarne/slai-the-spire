from dataclasses import dataclass

from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import get_actor_view
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.manager import ECSManager


@dataclass
class CharacterView(ActorView):
    pass


def get_character_view(manager: ECSManager) -> CharacterView:
    entity_id, _ = list(manager.get_component(CharacterComponent))[0]
    actor_view = get_actor_view(entity_id, manager)

    return CharacterView(
        actor_view.entity_id,
        actor_view.name,
        actor_view.health,
        actor_view.block,
        actor_view.modifiers,
    )
