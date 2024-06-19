from dataclasses import dataclass

from src.game.combat.view.actor import ActorView
from src.game.combat.view.actor import get_actor_view
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.manager import ECSManager


@dataclass
class MonsterView(ActorView):
    can_be_selected: bool
    # intents


def get_monster_view(entity_id: int, manager: ECSManager) -> MonsterView:
    actor_view = get_actor_view(entity_id, manager)
    can_be_selected_component = manager.get_component_for_entity(entity_id, CanBeSelectedComponent)

    return MonsterView(
        actor_view.entity_id,
        actor_view.name,
        actor_view.health,
        actor_view.block,
        actor_view.modifiers,
        False if can_be_selected_component is None else True,
    )


def get_monsters_view(manager: ECSManager) -> list[MonsterView]:
    # Create a list of tuples containing MonsterView objects and their positions
    monsters_view = [
        (get_monster_view(entity_id, manager), monster_component.position)
        for entity_id, monster_component in manager.get_component(MonsterComponent)
    ]

    # Sort the list of tuples by the position
    monsters_view.sort(key=lambda x: x[1])

    # Extract the sorted hand list
    monsters_view = [monster_view for monster_view, _ in monsters_view]

    return monsters_view
