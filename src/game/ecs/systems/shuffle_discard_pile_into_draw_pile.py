import random

from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import \
    ShuffleDiscardPileIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class ShuffleDiscardPileIntoDrawPileSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            shuffle_discard_pile_into_draw_pile_effect_component,
            effect_apply_to_component,
        ) = next(
            manager.get_components(
                ShuffleDiscardPileIntoDrawPileEffectComponent, EffectApplyToComponent
            )
        )
        # Get all cards in the discard and draw piles
        card_in_pile_entity_ids = effect_apply_to_component.entity_ids

        # Calculate shuffled positions
        positions = list(range(len(card_in_pile_entity_ids)))
        random.shuffle(positions)

        for card_in_pile_entity_id, position in zip(card_in_pile_entity_ids, positions):
            # Determine if the card is in the draw pile or in the discard pile
            card_in_draw_pile_component = manager.get_component_for_entity(
                card_in_pile_entity_id, CardInDrawPileComponent
            )
            if card_in_draw_pile_component is not None:
                # Card is in the draw pile. Shuffle its position
                card_in_draw_pile_component.position = position

            else:
                # Card is in the discard pile. Remove it from there and add it to the draw pile
                # in a random position
                manager.remove_component(card_in_pile_entity_id, CardInDiscardPileComponent)
                manager.add_component(card_in_pile_entity_id, CardInDrawPileComponent(position))

        return ProcessStatus.COMPLETE
