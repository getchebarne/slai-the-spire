from src.game.ecs.components.action import ActionComponent
from src.game.ecs.components.action import ActionConfirmComponent
from src.game.ecs.components.action import ActionEndTurnComponent
from src.game.ecs.components.action import ActionSelectComponent
from src.game.ecs.components.actors import BeforeTurnEndComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.cards import CardIsPlayedSingletonComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.effects import EffectInputTargetComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ProcessActionSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # Destroy previous card played event
        manager.destroy_component(CardIsPlayedSingletonComponent)

        try:
            action_entity_id, _ = next(manager.get_component(ActionComponent))

        except StopIteration:
            return

        if manager.get_component_for_entity(action_entity_id, ActionEndTurnComponent) is not None:
            # Get character
            character_entity_id, _ = next(manager.get_component(CharacterComponent))

            # Finish its turn and trigger its turn end. TODO: improve comment
            manager.remove_component(character_entity_id, IsTurnComponent)
            manager.add_component(character_entity_id, BeforeTurnEndComponent())
            manager.destroy_component(CardIsActiveComponent)

            return

        if manager.get_component_for_entity(action_entity_id, ActionConfirmComponent) is not None:
            # Get active card & tag it as played
            card_entity_id, _ = next(manager.get_components(CardIsActiveComponent))
            manager.add_component(card_entity_id, CardIsPlayedSingletonComponent())
            manager.destroy_component(CardIsActiveComponent)

            return

        if manager.get_component_for_entity(action_entity_id, ActionSelectComponent) is not None:
            for is_selected_entity_id, _ in manager.get_component(IsSelectedComponent):
                # If there's an effect pending input targets, tag the selected entity
                # TODO: support multiple selections
                if list(manager.get_component(EffectIsPendingInputTargetsComponent)):
                    manager.destroy_component(EffectInputTargetComponent)
                    manager.add_component(is_selected_entity_id, EffectInputTargetComponent())

                    return

                # If there's no effects pending input targets, and the selected entity is a card,
                # deactivate the previous card and activate the selected one
                if (
                    manager.get_component_for_entity(is_selected_entity_id, CardInHandComponent)
                    is not None
                ):
                    manager.destroy_component(CardIsActiveComponent)
                    manager.add_component(is_selected_entity_id, CardIsActiveComponent())

                # If there's no effects pending input targets, and the selected entity is a
                # monster, untarget the previous monster and target the selected one. The active
                # card is also tagged as played
                if (
                    manager.get_component_for_entity(is_selected_entity_id, MonsterComponent)
                    is not None
                ):
                    manager.destroy_component(CardTargetComponent)
                    manager.add_component(is_selected_entity_id, CardTargetComponent())

                    is_active_card_entity_id, _ = next(
                        manager.get_components(CardInHandComponent, CardIsActiveComponent)
                    )
                    manager.add_component(
                        is_active_card_entity_id, CardIsPlayedSingletonComponent()
                    )
                    manager.destroy_component(CardIsActiveComponent)
