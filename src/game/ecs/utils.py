from src.game.ecs.components.actors import ModifierParentActorComponent
from src.game.ecs.components.actors import ModifierStacksDurationComponent
from src.game.ecs.components.actors import TurnEndComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager


def add_effect_to_top(manager: ECSManager, effect_entity_id: int) -> None:
    for _, effect_is_queued_component in manager.get_component(EffectIsQueuedComponent):
        effect_is_queued_component.priority += 1

    manager.add_component(effect_entity_id, EffectIsQueuedComponent(0))


def add_effect_to_bot(manager: ECSManager, effect_entity_id: int) -> None:
    max_priority = -1
    for _, effect_is_queued_component in manager.get_component(EffectIsQueuedComponent):
        if effect_is_queued_component.priority > max_priority:
            max_priority = effect_is_queued_component.priority

    manager.add_component(effect_entity_id, EffectIsQueuedComponent(max_priority + 1))


def effect_queue_is_empty(manager: ECSManager) -> bool:
    return len(list(manager.get_component(EffectIsQueuedComponent))) == 0


# TODO: may need to reimplement as system
def trigger_actor_turn_end(manager: ECSManager, actor_entity_id: int) -> None:
    # Tag actor
    manager.add_component(actor_entity_id, TurnEndComponent())

    # Tag actor's modifiers
    for modifier_entity_id, modifier_parent_actor_component in manager.get_component(
        ModifierParentActorComponent
    ):
        if modifier_parent_actor_component.actor_entity_id == actor_entity_id:
            manager.add_component(modifier_entity_id, TurnEndComponent())

    # Create effect to decrease modifier stacks
    add_effect_to_bot(
        manager,
        manager.create_entity(
            EffectModifierDeltaComponent(-1),
            EffectQueryComponentsComponent([TurnEndComponent, ModifierStacksDurationComponent]),
        ),
    )
