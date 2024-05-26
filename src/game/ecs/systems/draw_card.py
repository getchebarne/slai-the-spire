from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import DrawCardEffectComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectToBeTargetedComponent
from src.game.ecs.components.effects import \
    ShuffleDiscardPileIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


MAX_HAND_SIZE = 10


class DrawCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (draw_card_effect_component, effect_apply_to_component) = next(
            manager.get_components(DrawCardEffectComponent, EffectApplyToComponent)
        )
        card_in_draw_pile_entity_ids = effect_apply_to_component.entity_ids
        hand_size = len(list(manager.get_components(CardInHandComponent)))

        # Sort cards in draw pile according to their position
        card_in_draw_pile_entity_ids.sort(
            key=lambda entity_id: manager.get_component_for_entity(
                entity_id, CardInDrawPileComponent
            ).position
        )
        # Draw cards
        num_cards_drawn = 0
        for _ in range(draw_card_effect_component.value):
            if len(card_in_draw_pile_entity_ids) == 0:
                break

            card_in_draw_pile_entity_id = card_in_draw_pile_entity_ids.pop(0)

            # Remove the card from the draw pile
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

        # Check if the draw pile ran out. If it did, create an effect to shuffle the discard pile
        # into the draw pile and another effect to draw the remaining cards
        if (
            len(card_in_draw_pile_entity_ids) == 0
            and num_cards_drawn < draw_card_effect_component.value
        ):
            for effect_entity_id, effect_to_be_targeted_component in manager.get_component(
                EffectToBeTargetedComponent
            ):
                effect_to_be_targeted_component.priority += 2

            manager.create_entity(
                ShuffleDiscardPileIntoDrawPileEffectComponent(),
                EffectQueryComponentsComponent([CardInDiscardPileComponent]),
                EffectSelectionTypeComponent(EffectSelectionType.ALL),
                EffectToBeTargetedComponent(priority=0),
            )
            manager.create_entity(
                DrawCardEffectComponent(draw_card_effect_component.value - num_cards_drawn),
                EffectQueryComponentsComponent([CardInDrawPileComponent]),
                EffectSelectionTypeComponent(EffectSelectionType.ALL),
                EffectToBeTargetedComponent(priority=1),
            )
            return ProcessStatus.INCOMPLETE

        # Update the positions of cards in the draw pile based on the number of cards drawn
        for card_in_draw_pile_entity_id, card_in_draw_pile_component in manager.get_component(
            CardInDrawPileComponent
        ):
            card_in_draw_pile_component.position -= num_cards_drawn

        return ProcessStatus.COMPLETE
