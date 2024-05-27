from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import DiscardHandAtEndOfTurnEffect
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.discard_hand_at_end_of_turn import DiscardHandAtEndOfTurnSystem


def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create cards in hand
    num_cards = 5
    card_in_hand_entity_ids = [
        manager.create_entity(CardInHandComponent(i)) for i in range(num_cards)
    ]

    # Create effect
    manager.create_entity(DiscardHandAtEndOfTurnEffect(), EffectIsDispatchedComponent())

    # Run the system
    DiscardHandAtEndOfTurnSystem().process(manager)

    # Verify the hand is empty
    assert len(list(manager.get_component(CardInHandComponent))) == 0

    # Verify all cards previously in the hand are now in the discard pile
    for card_in_hand_entity_id in card_in_hand_entity_ids:
        assert (
            manager.get_component_for_entity(card_in_hand_entity_id, CardInDiscardPileComponent)
            is not None
        )


def test_empty_hand() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create some cards in the discard pile
    num_cards = 5
    card_in_discard_pile_entity_ids = [
        manager.create_entity(CardInDiscardPileComponent()) for i in range(num_cards)
    ]

    # Create effect
    manager.create_entity(DiscardHandAtEndOfTurnEffect(), EffectIsDispatchedComponent())

    # Run the system
    DiscardHandAtEndOfTurnSystem().process(manager)

    # Verify the hand is still empty
    assert len(list(manager.get_component(CardInHandComponent))) == 0

    # Verify the discard pile hasn't changed
    for card_in_discard_pile_entity_id in card_in_discard_pile_entity_ids:
        assert (
            manager.get_component_for_entity(
                card_in_discard_pile_entity_id, CardInDiscardPileComponent
            )
            is not None
        )
