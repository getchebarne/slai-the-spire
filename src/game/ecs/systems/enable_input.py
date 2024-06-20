from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.cards import CardRequiresTargetComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.effects import EffectIsPendingInputTargetsComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: revisit if this needs to return if it's not the character's turn
class EnableInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # Disable all inputs to begin with
        manager.destroy_component(CanBeSelectedComponent)

        # If there's an effect pending input targets (e.g., discard card), its query entities can
        # be selected
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

        # If there's no effects pending input targets, all cards in hand w/ cost less or equal to
        # the player's energy can be selected
        _, energy_component = next(manager.get_component(EnergyComponent))
        for card_in_hand_entity_id, (card_cost_component, _) in manager.get_components(
            CardCostComponent, CardInHandComponent
        ):
            if card_cost_component.value <= energy_component.current:
                manager.add_component(card_in_hand_entity_id, CanBeSelectedComponent())

        # If there's a card selected in hand that requires a target, monsters can be selected
        # TODO: fix, cardisactive shouldn't remain after the card is played
        query_result = list(manager.get_component(CardIsActiveComponent))
        if query_result:
            card_is_selected_entity_id, _ = query_result[0]

            if manager.get_component_for_entity(
                card_is_selected_entity_id, CardRequiresTargetComponent
            ):
                for monster_entity_id, _ in manager.get_component(MonsterComponent):
                    manager.add_component(monster_entity_id, CanBeSelectedComponent())
