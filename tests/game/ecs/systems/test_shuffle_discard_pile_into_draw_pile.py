from src.game.ecs.components.cards import CardInDiscardPileComponent
from src.game.ecs.components.cards import CardInDrawPileComponent
from src.game.ecs.components.cards import CardInPileComponent
from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.components.effects import \
    ShuffleDiscardPileIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.shuffle_discard_pile_into_draw_pile import \
    ShuffleDiscardPileIntoDrawPileSystem


# TODO: change number of cards in draw and discard piles
def test_base() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_cards` cards in the draw pile and `num_cards` cards in the discard pile
    num_cards = 5
    card_in_draw_pile_entity_ids = [
        manager.create_entity(CardInPileComponent(), CardInDrawPileComponent(i))
        for i in range(num_cards)
    ]
    card_in_discard_pile_entity_ids = [
        manager.create_entity(CardInPileComponent(), CardInDiscardPileComponent())
        for i in range(num_cards)
    ]
    card_in_pile_entity_ids = card_in_draw_pile_entity_ids + card_in_discard_pile_entity_ids

    # Create effect to shuffle the discard pile into the draw pile
    # TODO: should it be passed through the targeting system?
    manager.create_entity(
        ShuffleDiscardPileIntoDrawPileEffectComponent(),
        EffectApplyToComponent(card_in_pile_entity_ids),
    )
    # Run the system
    ShuffleDiscardPileIntoDrawPileSystem().process(manager)

    # Assert there's no cards in the discard pile
    assert len(list(manager.get_component(CardInDiscardPileComponent))) == 0

    # Assert all cards in a pile are in the discard pile
    card_in_pile_entity_ids = [
        card_in_pile_entity_id
        for card_in_pile_entity_id, _ in manager.get_component(CardInPileComponent)
    ]
    card_in_draw_pile_entity_ids = [
        card_in_draw_pile_entity_id
        for card_in_draw_pile_entity_id, _ in manager.get_component(CardInPileComponent)
    ]
    assert sorted(card_in_pile_entity_ids) == sorted(card_in_draw_pile_entity_ids)

    # Assert the card's positions range from 0 to 2 * `num_cards` - 1
    card_in_draw_pile_positions = [
        card_in_draw_pile_component.position
        for _, card_in_draw_pile_component in manager.get_component(CardInDrawPileComponent)
    ]
    assert sorted(card_in_draw_pile_positions) == list(range(2 * num_cards))


def test_empty_discard_pile() -> None:
    # Instance ECS manager
    manager = ECSManager()

    # Create `num_cards` cards in the draw pile
    num_cards = 10
    card_in_draw_pile_entity_ids = [
        manager.create_entity(CardInPileComponent(), CardInDrawPileComponent(i))
        for i in range(num_cards)
    ]
    prev_positions = list(range(num_cards))

    # Create effect to shuffle the discard pile into the draw pile
    # TODO: should these be passed through the targeting system?
    manager.create_entity(
        ShuffleDiscardPileIntoDrawPileEffectComponent(),
        EffectApplyToComponent(card_in_draw_pile_entity_ids),
    )
    # Run system
    ShuffleDiscardPileIntoDrawPileSystem().process(manager)

    # Assert the positions have been shuffled. This can fail in case the positions stay the
    # same, but the probability should be practically zero if `num_cards` is sufficiently large
    next_positions = [
        manager.get_component_for_entity(
            card_in_draw_pile_entity_id, CardInDrawPileComponent
        ).position
        for card_in_draw_pile_entity_id in card_in_draw_pile_entity_ids
    ]
    assert len(next_positions) == num_cards
    assert next_positions != prev_positions
