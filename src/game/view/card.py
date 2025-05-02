from dataclasses import dataclass

from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager
from src.game.utils import does_card_require_target
from src.game.view.effect import ViewEffect


@dataclass(frozen=True)
class ViewCard:
    name: str
    color: CardColor
    type: CardType
    rarity: CardRarity
    cost: int
    effects: list[ViewEffect]
    exhaust: bool
    is_active: bool
    requires_target: bool


def _get_card_view(card: EntityCard, is_active: bool) -> ViewCard:
    return ViewCard(
        card.name,
        card.color,
        card.type,
        card.rarity,
        card.cost,
        card.effects,
        card.exhaust,
        is_active,
        does_card_require_target(card),
    )


def get_view_hand(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_cards_in_hand:
        card = entity_manager.entities[id_card]
        is_active = id_card == entity_manager.id_card_active
        card_views.append(_get_card_view(card, is_active))

    return card_views


def get_view_pile_draw(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_cards_in_draw_pile:
        if id_card == entity_manager.id_card_active:
            raise ValueError(f"Card with id {id_card} is in the draw pile but it's active")

        card = entity_manager.entities[id_card]
        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_pile_disc(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_cards_in_disc_pile:
        if id_card == entity_manager.id_card_active:
            raise ValueError(f"Card with id {id_card} is in the discard pile but it's active")

        card = entity_manager.entities[id_card]
        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_pile_exhaust(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_cards_in_exhaust_pile:
        if id_card == entity_manager.id_card_active:
            raise ValueError(f"Card with id {id_card} is in the exhaust pile but it's active")

        card = entity_manager.entities[id_card]
        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_deck(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_cards_in_deck:
        card = entity_manager.entities[id_card]
        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_reward_combat(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for id_card in entity_manager.id_card_reward:
        if id_card == entity_manager.id_card_active:
            raise ValueError(f"Card with id {id_card} is in combat rewards but it's active")

        card = entity_manager.entities[id_card]
        card_views.append(_get_card_view(card, False))

    return card_views
