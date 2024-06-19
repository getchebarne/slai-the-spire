from src.game.ecs.components.effects import EffectIsQueuedComponent
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
