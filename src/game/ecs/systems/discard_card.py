from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: can this be faster? maybe the sort slows things down
class DiscardCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, (draw_card_effect_component, _) = next(
                manager.get_components(
                    EffectDiscardCardComponent, EffectIsTargetedSingletonComponent
                )
            )

        except StopIteration:
            return

        # Get target entities
        target_entity_ids = [
            target_entity_id
            for target_entity_id, _ in manager.get_component(EffectTargetComponent)
        ]
        # Get cards in hand and sort them according to their position
        card_in_hands = list(manager.get_component(CardInHandComponent))
        card_in_hands.sort(key=lambda card_in_hand: card_in_hand[1].position)

        # Iterate over cards to discard
        for target_entity_id in target_entity_ids:
            position = manager.get_component_for_entity(
                target_entity_id, CardInHandComponent
            ).position

            # Shift cards in hand to the left
            for card_in_hand_entity_id, card_in_hand_component in card_in_hands[position + 1 :]:
                card_in_hand_component.position -= 1

            # Move card from hand to discard pile
            manager.remove_component(target_entity_id, CardInHandComponent)
            manager.add_component(target_entity_id, CardInDiscardPileComponent())
