from dataclasses import dataclass

from src.game.combat.entities import Effect
from src.game.combat.entities import EntityManager


@dataclass
class CardView:
    entity_id: int
    name: str
    effects: list[Effect]  # TODO: implement EffectView
    cost: int
    is_active: bool


def _card_to_view(entity_manager: EntityManager, id_card: int) -> CardView:
    card = entity_manager.entities[id_card]

    return CardView(
        id_card,
        card.name,
        card.effects,
        card.cost,
        True if id_card == entity_manager.id_card_active else False,
    )


def view_draw_pile(entity_manager: EntityManager) -> list[CardView]:
    return [
        _card_to_view(entity_manager, id_card_in_draw_pile)
        for id_card_in_draw_pile in entity_manager.id_cards_in_draw_pile
    ]


def view_hand(entity_manager: EntityManager) -> list[CardView]:
    return [
        _card_to_view(entity_manager, id_card_in_hand)
        for id_card_in_hand in entity_manager.id_cards_in_hand
    ]


def view_discard_pile(entity_manager: EntityManager) -> set[CardView]:
    return [
        _card_to_view(entity_manager, id_card_in_disc_pile)
        for id_card_in_disc_pile in entity_manager.id_cards_in_disc_pile
    ]
