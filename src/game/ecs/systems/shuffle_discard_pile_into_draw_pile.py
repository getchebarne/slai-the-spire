import random

from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import EffectShuffleDiscardPileIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ShuffleDiscardPileIntoDrawPileSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, (shuffle_discard_pile_into_draw_pile_effect_component, _) = next(
                manager.get_components(
                    EffectShuffleDiscardPileIntoDrawPileComponent, EffectIsTargetedComponent
                )
            )

        except StopIteration:
            return

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
            card_in_discard_piles, positions[len(card_in_draw_piles) :]
        ):
            manager.remove_component(card_in_discard_pile_entity_id, CardInDiscardPileComponent)
            manager.add_component(
                card_in_discard_pile_entity_id, CardInDrawPileComponent(position)
            )

        return
