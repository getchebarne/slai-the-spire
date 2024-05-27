import random

from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import ShuffleDiscardPileIntoDrawPileEffectComponent
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
                ShuffleDiscardPileIntoDrawPileEffectComponent, EffectIsDispatchedComponent
            )
        )
        # Get all cards in the discard and draw piles
        card_in_draw_piles = list(manager.get_component(CardInDrawPileComponent))
        card_in_discard_piles = list(manager.get_component(CardInDiscardPileComponent))

        # Calculate shuffled positions
        positions = list(range(len(card_in_draw_piles + card_in_discard_piles)))
        random.shuffle(positions)

        # Shuffle cards in the draw pile
        for (card_in_draw_pile_entity_id, card_in_draw_pile_component), position in zip(
            card_in_draw_piles, positions[: len(card_in_draw_piles)]
        ):
            card_in_draw_pile_component.position = position

        # Shuffle cards in the discard pile
        for (card_in_discard_pile_entity_id, card_in_discard_pile_component), position in zip(
            card_in_discard_piles, positions[len(card_in_discard_piles) :]
        ):
            manager.remove_component(card_in_discard_pile_entity_id, CardInDiscardPileComponent)
            manager.add_component(
                card_in_discard_pile_entity_id, CardInDrawPileComponent(position)
            )

        return ProcessStatus.COMPLETE
