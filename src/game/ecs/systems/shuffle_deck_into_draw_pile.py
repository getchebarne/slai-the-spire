import random

from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInPileComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import \
    ShuffleDeckIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class ShuffleDeckIntoDrawPileSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (
            shuffle_deck_into_draw_pile_effect_component,
            effect_apply_to_component,
        ) = next(
            manager.get_components(ShuffleDeckIntoDrawPileEffectComponent, EffectApplyToComponent)
        )
        # Get all cards in the deck
        card_in_deck_entity_ids = effect_apply_to_component.entity_ids

        # Calculate shuffled positions
        positions = list(range(len(card_in_deck_entity_ids)))
        random.shuffle(positions)

        # Add each card in the deck to the draw pile in a random position
        for card_in_deck_entity_id, position in zip(card_in_deck_entity_ids, positions):
            manager.add_component(card_in_deck_entity_id, CardInPileComponent())
            manager.add_component(card_in_deck_entity_id, CardInDrawPileComponent(position))

        # Untag effect
        manager.remove_component(effect_entity_id, EffectApplyToComponent)

        return ProcessStatus.COMPLETE
