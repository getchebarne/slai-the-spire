from src.game.ecs.components.base import BaseComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.manager import ECSManager


def add_effect_to_top(manager: ECSManager, *components: BaseComponent) -> None:
    for _, effect_is_queued_component in manager.get_component(EffectIsQueuedComponent):
        effect_is_queued_component.priority += 1

    manager.create_entity(*components, EffectIsQueuedComponent(0))


def add_effect_to_bot(manager: ECSManager, *components: BaseComponent) -> None:
    max_priority = None
    for _, effect_is_queued_component in manager.get_component(EffectIsQueuedComponent):
        if max_priority is None:
            max_priority = effect_is_queued_component.priority
            continue

        if effect_is_queued_component.priority > max_priority:
            max_priority = effect_is_queued_component.priority

    manager.create_entity(*components, EffectIsQueuedComponent(max_priority + 1))
