from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


def _card_requires_target(manager: ECSManager, card_entity_id: int) -> bool:
    for effect_entity_id in manager.get_component_for_entity(
        card_entity_id, CardHasEffectsComponent
    ).effect_entity_ids:
        effect_query_components_component = manager.get_component_for_entity(
            effect_entity_id, EffectQueryComponentsComponent
        )
        if CardTargetComponent in effect_query_components_component.value:
            return True

    return False


class EnableInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # Disable all inputs to begin with
        manager.destroy_component(CanBeSelectedComponent)

        # If it's not the character's turn, return
        if len(list(manager.get_components(CharacterComponent, IsTurnComponent))) == 0:
            return

        # If there's queued effects, return
        if len(list(manager.get_component(EffectIsQueuedComponent))) > 0:
            return

        # If there's an effect pending input targets, mark its query entities as selectable, then
        # return
        query_result = list(
            manager.get_components(
                EffectQueryComponentsComponent, EffectIsPendingInputTargetsComponent
            )
        )
        if query_result:
            _, (effect_query_components_component, _) = query_result[0]
            for query_entity_id, _ in manager.get_components(
                *effect_query_components_component.value
            ):
                manager.add_component(query_entity_id, CanBeSelectedComponent())

            return

        # At this point, if it's the character's turn, there's no queued effects, and there's no
        # effects pending input targets, all cards in hand are selectable
        for card_in_hand_entity_id, _ in manager.get_component(CardInHandComponent):
            manager.add_component(card_in_hand_entity_id, CanBeSelectedComponent())

        # If there's a card selected in hand that requires a target, monsters are also selectable
        query_result = list(manager.get_components(CardInHandComponent, IsSelectedComponent))
        if query_result:
            card_is_selected_entity_id, _ = query_result[0]

            if _card_requires_target(manager, card_is_selected_entity_id):
                for monster_entity_id, _ in manager.get_component(MonsterComponent):
                    manager.add_component(monster_entity_id, CanBeSelectedComponent())
