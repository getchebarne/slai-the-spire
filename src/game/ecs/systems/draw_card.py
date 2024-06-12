from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectShuffleDiscardPileIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_top


MAX_HAND_SIZE = 10


class DrawCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, (draw_card_effect_component, _) = next(
                manager.get_components(EffectDrawCardComponent, EffectIsTargetedComponent)
            )

        except StopIteration:
            return

        # Get cards in draw pile and sort them according to their position
        card_in_discard_piles = list(manager.get_component(CardInDiscardPileComponent))
        card_in_draw_piles = list(manager.get_component(CardInDrawPileComponent))
        if len(card_in_discard_piles) == 0 and len(card_in_draw_piles) == 0:
            return

        card_in_draw_piles.sort(key=lambda x: x[1].position)
        hand_size = len(list(manager.get_components(CardInHandComponent)))

        # Draw cards
        num_cards_drawn = 0
        for _ in range(draw_card_effect_component.value):
            if len(card_in_draw_piles) == 0:
                break

            # Remove the card from the draw pile
            card_in_draw_pile_entity_id = card_in_draw_piles.pop(0)[0]
            manager.remove_component(card_in_draw_pile_entity_id, CardInDrawPileComponent)
            num_cards_drawn += 1

            # Draw card either to the draw pile or discard pile
            if hand_size >= MAX_HAND_SIZE:
                manager.add_component(card_in_draw_pile_entity_id, CardInDiscardPileComponent())

            else:
                manager.add_component(
                    card_in_draw_pile_entity_id, CardInHandComponent(position=hand_size)
                )
                hand_size += 1

        # Update the positions of cards in the draw pile based on the number of cards drawn
        for card_in_draw_pile_entity_id, card_in_draw_pile_component in manager.get_component(
            CardInDrawPileComponent
        ):
            card_in_draw_pile_component.position -= num_cards_drawn

        # Check if the draw pile ran out. If it did, create an effect to shuffle the discard pile
        # into the draw pile and another effect to draw the remaining cards
        if len(card_in_draw_piles) == 0 and num_cards_drawn < draw_card_effect_component.value:
            add_effect_to_top(
                manager,
                EffectDrawCardComponent(draw_card_effect_component.value - num_cards_drawn),
            )
            add_effect_to_top(manager, EffectShuffleDiscardPileIntoDrawPileComponent())
            return
