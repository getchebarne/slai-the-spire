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


def _card_requires_target(card_entity_id: int, manager: ECSManager) -> bool:
    for effect_entity_id in manager.get_component_for_entity(
        card_entity_id, CardHasEffectsComponent
    ).effect_entity_ids:
        effect_query_components_component = manager.get_component_for_entity(
            effect_entity_id, EffectQueryComponentsComponent
        )
        if CardTargetComponent in effect_query_components_component.value:
            return True

    return False


# TODO: improve this, it's a bit bad
class EnableInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        manager.destroy_component(CanBeSelectedComponent)

        try:
            # If it's not the character's turn, disable input
            next(manager.get_components(CharacterComponent, IsTurnComponent))

        except StopIteration:
            manager.destroy_component(CanBeSelectedComponent)

            return

        try:
            # If there's an effect pending input targets, mark its query entities as selectable
            _, (_, effect_query_components_component) = next(
                manager.get_components(
                    EffectIsPendingInputTargetsComponent, EffectQueryComponentsComponent
                )
            )
            for query_entity_id, _ in manager.get_components(
                *effect_query_components_component.value
            ):
                manager.add_component(query_entity_id, CanBeSelectedComponent())

            return

        except StopIteration:
            pass

        # If there's a card selected in the hand, all other cards are also selectable. If the
        # selected card requires a target, monsters should also be selectable
        try:
            card_is_selected_entity_id, _ = next(
                manager.get_components(CardInHandComponent, IsSelectedComponent)
            )
            for card_in_hand_entity_id, _ in manager.get_component(CardInHandComponent):
                manager.add_component(card_in_hand_entity_id, CanBeSelectedComponent())

            if _card_requires_target(card_is_selected_entity_id, manager):
                for monster_entity_id, _ in manager.get_component(MonsterComponent):
                    manager.add_component(monster_entity_id, CanBeSelectedComponent())

        except StopIteration:
            pass

        # Else, if there's queued effects, disable all inputs
        if len(list(manager.get_component(EffectIsQueuedComponent))) > 0:
            manager.destroy_component(CanBeSelectedComponent)

            return

        # Otherwise, we're in "default" state. Make all cards selectable
        for card_in_hand_entity_id, _ in manager.get_component(CardInHandComponent):
            manager.add_component(card_in_hand_entity_id, CanBeSelectedComponent())
