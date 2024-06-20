from dataclasses import dataclass

from src.game.combat.view.modifier import ModifierView
from src.game.combat.view.modifier import get_modifier_view
from src.game.ecs.components.actors import BlockComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import ModifierParentComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


@dataclass
class HealthView:
    current: int
    max: int


@dataclass
class BlockView:
    current: int


@dataclass
class ActorView:
    entity_id: int
    name: str
    health: HealthView
    block: BlockView
    modifiers: list[ModifierView]
    is_turn: bool


def get_actor_view(entity_id: int, manager: ECSManager) -> ActorView:
    name_component = manager.get_component_for_entity(entity_id, NameComponent)
    health_component = manager.get_component_for_entity(entity_id, HealthComponent)
    block_component = manager.get_component_for_entity(entity_id, BlockComponent)
    is_turn_component = manager.get_component_for_entity(entity_id, IsTurnComponent)

    modifier_views = _get_modifier_views_for_actor(entity_id, manager)

    return ActorView(
        entity_id,
        name_component.value,
        HealthView(health_component.current, health_component.max),
        BlockView(block_component.current),
        modifier_views,
        False if is_turn_component is None else True,
    )


def _get_modifier_views_for_actor(entity_id: int, manager: ECSManager) -> list[ModifierView]:
    return [
        get_modifier_view(modifier_entity_id, manager)
        for modifier_entity_id, modifier_parent_component in manager.get_component(
            ModifierParentComponent
        )
        if modifier_parent_component.actor_entity_id == entity_id
    ]
