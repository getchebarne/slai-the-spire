from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import DiscardHandAtEndOfTurnEffect
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class DiscardHandAtEndOfTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        try:
            _ = next(
                manager.get_components(DiscardHandAtEndOfTurnEffect, EffectIsDispatchedComponent)
            )

        except StopIteration:
            return ProcessStatus.PASS

        # Get all cards in the hand
        card_in_hand_entity_ids = [
            card_in_hand_entity_id
            for card_in_hand_entity_id, _ in manager.get_component(CardInHandComponent)
        ]

        # Move them to the discard pile
        for card_in_hand_entity_id in card_in_hand_entity_ids:
            manager.remove_component(card_in_hand_entity_id, CardInHandComponent)
            manager.add_component(card_in_hand_entity_id, CardInDiscardPileComponent())

        return ProcessStatus.COMPLETE
