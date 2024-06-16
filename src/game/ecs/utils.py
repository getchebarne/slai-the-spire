from src.game.ecs.components.actors import ActorHasModifiersComponent
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
    actor_has_modifiers_component = manager.get_component_for_entity(
        actor_entity_id, ActorHasModifiersComponent
    )
    if actor_has_modifiers_component is not None:
        for modifier_entity_id in actor_has_modifiers_component.modifier_entity_ids:
            manager.add_component(modifier_entity_id, TurnEndComponent())

        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectModifierDeltaComponent(-1),
                EffectQueryComponentsComponent(
                    [TurnEndComponent, ModifierStacksDurationComponent]
                ),
            ),
        )
