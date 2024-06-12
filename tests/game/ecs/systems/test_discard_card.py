from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.discard_card import DiscardCardSystem


# TODO: expand test cases
def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_cards` in hand
    num_cards = 5
    card_in_hand_entity_ids = [
        manager.create_entity(CardInHandComponent(i)) for i in range(num_cards)
    ]

    # Create effect to shuffle the discard pile into the draw pile
    manager.create_entity(EffectDiscardCardComponent(), EffectIsDispatchedComponent())

    # Set a card as the target of the discard effect
    target_entity_id = card_in_hand_entity_ids[len(card_in_hand_entity_ids) // 2]
    manager.add_component(target_entity_id, EffectTargetComponent())

    # Run the system
    DiscardCardSystem().process(manager)

    # Assert the target card has been discarded
    assert manager.get_component_for_entity(target_entity_id, CardInHandComponent) is None
    assert target_entity_id in [
        card_in_discard_pile_entity_id
        for card_in_discard_pile_entity_id, _ in manager.get_component(CardInDiscardPileComponent)
    ]

    target_entity_idx = card_in_hand_entity_ids.index(target_entity_id)
    for position, card_in_hand_entity_id in enumerate(card_in_hand_entity_ids):
        if position < target_entity_idx:
            # Assert the cards to the left of the discarded card are in the same position
            assert (
                position
                == manager.get_component_for_entity(
                    card_in_hand_entity_id, CardInHandComponent
                ).position
            )

        elif position > target_entity_idx:
            # Assert the cards to the right of the discarded card have been shifted to the left
            assert (
                position
                == manager.get_component_for_entity(
                    card_in_hand_entity_id, CardInHandComponent
                ).position
                + 1
            )
