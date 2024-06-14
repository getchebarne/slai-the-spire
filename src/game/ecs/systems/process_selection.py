from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import EffectInputTargetComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: change name
class ProcessSelectionSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            is_selected_entity_id, _ = next(manager.get_component(IsSelectedComponent))

        except StopIteration:
            return

        # If there's an effect pending input targets, tag the selected entity an return
        # TODO: support multiple entity selection
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

            return

        # If there's no effects pending input targets, and the selected entity is a monster,
        # untarget the previous monster and target the selected one. The active card is also
        # tagged as played
        if manager.get_component_for_entity(is_selected_entity_id, MonsterComponent) is not None:
            manager.destroy_component(CardTargetComponent)
            manager.add_component(is_selected_entity_id, CardTargetComponent())

            is_active_card_entity_id, _ = next(
                manager.get_components(CardInHandComponent, CardIsActiveComponent)
            )
            manager.add_component(is_active_card_entity_id, CardIsPlayedComponent())
