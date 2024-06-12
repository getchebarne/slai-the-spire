from src.game.ecs.components.cards import CardInDeckComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectShuffleDeckIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.shuffle_deck_into_draw_pile import ShuffleDeckIntoDrawPileSystem


# TODO: test w/ different seeds
def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create a starter deck
    num_cards = 10
    card_in_deck_entity_ids = [
        manager.create_entity(CardInDeckComponent()) for _ in range(num_cards)
    ]

    # Create effect to shuffle the deck into the draw pile
    # TODO: should it be passed through the targeting system?
    manager.create_entity(EffectShuffleDeckIntoDrawPileComponent(), EffectIsDispatchedComponent())

    # Run the system
    ShuffleDeckIntoDrawPileSystem().process(manager)

    # Get all cards in draw pile
    card_in_draw_pile_entity_ids = []
    positions = []
    for card_in_draw_pile_entity_id, card_in_draw_pile_component in manager.get_component(
        CardInDrawPileComponent
    ):
        card_in_draw_pile_entity_ids.append(card_in_draw_pile_entity_id)
        positions.append(card_in_draw_pile_component.position)

    # Assert all cards in deck are in draw pile
    assert sorted(card_in_deck_entity_ids) == sorted(card_in_draw_pile_entity_ids)

    # Assert the cards' positions range from 0 to `len(card_in_deck_entity_ids)`
    assert sorted(positions) == list(range(len(card_in_deck_entity_ids)))

    # Assert the cards are still tagged as part of the deck
    for card_in_deck_entity_id in card_in_deck_entity_ids:
        assert (
            manager.get_component_for_entity(card_in_deck_entity_id, CardInDeckComponent)
            is not None
        )
