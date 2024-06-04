import random

from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import ShuffleDeckIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ShuffleDeckIntoDrawPileSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            effect_entity_id, (shuffle_deck_into_draw_pile_effect_component, _) = next(
                manager.get_components(
                    ShuffleDeckIntoDrawPileEffectComponent, EffectIsTargetedComponent
                )
            )

        except StopIteration:
            return

        # Get all cards in the deck
        card_in_decks = list(manager.get_component(CardInDeckComponent))

        # Shuffle positions
        random.shuffle(card_in_decks)

        # Add each card in the deck to the draw pile in a random position
        for position, (card_in_deck_entity_id, _) in enumerate(card_in_decks):
            manager.add_component(card_in_deck_entity_id, CardInDrawPileComponent(position))

        return
