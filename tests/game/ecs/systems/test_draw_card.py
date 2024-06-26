from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectIsDispatchedComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectShuffleDiscardPileIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.draw_card import DrawCardSystem


# TODO: change number of cards in draw and discard piles
def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_cards` cards in draw pile and `num_cards` cards in hand
    num_cards = 5
    card_in_draw_pile_entity_ids = [
        manager.create_entity(CardInDrawPileComponent(i)) for i in range(num_cards)
    ]
    card_in_hand_entity_ids = [
        manager.create_entity(CardInHandComponent(i)) for i in range(num_cards)
    ]

    # Create effect to shuffle the discard pile into the draw pile
    # TODO: should it be passed through the targeting system?
    num_draw = 3
    manager.create_entity(EffectDrawCardComponent(num_draw), EffectIsDispatchedComponent())

    # Run the system
    DrawCardSystem().process(manager)

    # Assert the previous cards in hand are still there and in the same position
    for i, card_in_hand_entity_id in enumerate(card_in_hand_entity_ids):
        assert (
            manager.get_component_for_entity(card_in_hand_entity_id, CardInHandComponent).position
            == i
        )

    # Assert the top `num_draw` cards from the draw pile are now in hand and that their
    # position ranges from `num_cards` to `num_cards` + `num_draw` - 1
    for i, card_in_draw_pile_entity_id in enumerate(card_in_draw_pile_entity_ids[:num_draw]):
        assert (
            manager.get_component_for_entity(
                card_in_draw_pile_entity_id, CardInHandComponent
            ).position
            == num_cards + i
        )

    # Assert the rest of the cards are still in draw pile, and that their position has been
    # updated correctly
    for i, card_in_draw_pile_entity_id in enumerate(card_in_draw_pile_entity_ids[num_draw:]):
        assert (
            manager.get_component_for_entity(
                card_in_draw_pile_entity_id, CardInDrawPileComponent
            ).position
            == i
        )


def test_unsufficient_cards_in_draw_pile() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_cards` cards in draw pile and `num_cards` cards in hand
    num_cards = 2
    card_in_draw_pile_entity_ids = [
        manager.create_entity(CardInDrawPileComponent(i)) for i in range(num_cards)
    ]
    card_in_hand_entity_ids = [
        manager.create_entity(CardInHandComponent(i)) for i in range(num_cards)
    ]

    # Create effect to shuffle the discard pile into the draw pile
    # TODO: should it be passed through the targeting system?
    num_draw = 3
    manager.create_entity(EffectDrawCardComponent(num_draw), EffectIsDispatchedComponent())

    # Run the system
    DrawCardSystem().process(manager)

    # Assert the previous cards in hand are still there and in the same position
    for i, card_in_hand_entity_id in enumerate(card_in_hand_entity_ids):
        assert (
            manager.get_component_for_entity(card_in_hand_entity_id, CardInHandComponent).position
            == i
        )

    # Assert all cards previously in draw pile are now in hand
    for i, card_in_draw_pile_entity_id in enumerate(card_in_draw_pile_entity_ids):
        assert (
            manager.get_component_for_entity(
                card_in_draw_pile_entity_id, CardInHandComponent
            ).position
            == num_cards + i
        )

    # Assert the draw pile is empty
    assert len(list(manager.get_component(CardInDrawPileComponent))) == 0

    # Assert one and only one effect to shuffle the discard pile into the draw pile has been
    # created w/ position 0
    query_result = list(
        manager.get_components(
            EffectShuffleDiscardPileIntoDrawPileComponent, EffectIsQueuedComponent
        )
    )
    assert len(query_result) == 1

    _, (_, effect_to_be_targeted_component) = query_result[0]
    assert effect_to_be_targeted_component.position == 0

    # Assert one and only one effect to draw the remaining cards has been created w/ position 1
    query_result = list(manager.get_components(EffectDrawCardComponent, EffectIsQueuedComponent))
    assert len(query_result) == 1

    _, (draw_card_effect_component, effect_to_be_targeted_component) = query_result[0]
    assert effect_to_be_targeted_component.position == 1
    assert draw_card_effect_component.value == num_draw - num_cards
