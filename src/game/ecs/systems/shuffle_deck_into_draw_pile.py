import random

from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import ShuffleDeckIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class ShuffleDeckIntoDrawPileSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        try:
            effect_entity_id, (
                shuffle_deck_into_draw_pile_effect_component,
                effect_apply_to_component,
            ) = next(
                manager.get_components(
                    ShuffleDeckIntoDrawPileEffectComponent, EffectIsDispatchedComponent
                )
            )

        except StopIteration:
            return ProcessStatus.PASS

        # Get all cards in the deck
        card_in_decks = list(manager.get_component(CardInDeckComponent))

        # Shuffle positions
        random.shuffle(card_in_decks)

        # Add each card in the deck to the draw pile in a random position
        for position, (card_in_deck_entity_id, _) in enumerate(card_in_decks):
            manager.add_component(card_in_deck_entity_id, CardInDrawPileComponent(position))

        return ProcessStatus.COMPLETE
