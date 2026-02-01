from dataclasses import dataclass

from src.game.entity.card import CardColor
from src.game.entity.card import CardRarity
from src.game.entity.card import CardType
from src.game.entity.card import EntityCard
from src.game.entity.manager import EntityManager
from src.game.utils import does_card_require_discard
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
    innate: bool
    is_active: bool
    requires_target: bool
    requires_discard: bool


def _get_card_view(card: EntityCard, is_active: bool) -> ViewCard:
    return ViewCard(
        card.name,
        card.color,
        card.type,
        card.rarity,
        card.cost,
        card.effects,
        card.exhaust,
        card.innate,
        is_active,
        does_card_require_target(card),
        does_card_require_discard(card),
    )


def get_view_hand(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.hand:
        is_active = card is entity_manager.card_active
        card_views.append(_get_card_view(card, is_active))

    return card_views


def get_view_pile_draw(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.draw_pile:
        if card is entity_manager.card_active:
            raise ValueError(f"Card {card.name} is in the draw pile but it's active")

        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_pile_disc(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.disc_pile:
        if card is entity_manager.card_active:
            raise ValueError(f"Card {card.name} is in the discard pile but it's active")

        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_pile_exhaust(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.exhaust_pile:
        if card is entity_manager.card_active:
            raise ValueError(f"Card {card.name} is in the exhaust pile but it's active")

        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_deck(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.deck:
        card_views.append(_get_card_view(card, False))

    return card_views


def get_view_reward_combat(entity_manager: EntityManager) -> list[ViewCard]:
    card_views = []
    for card in entity_manager.card_reward:
        if card is entity_manager.card_active:
            raise ValueError(f"Card {card.name} is in combat rewards but it's active")

        card_views.append(_get_card_view(card, False))

    return card_views
