from dataclasses import dataclass

from src.game.combat.context import Card
from src.game.combat.context import GameContext


@dataclass
class CardView:
    name: str
    cost: int
    is_active: bool  # TODO: make int

    def __hash__(self) -> int:
        return hash(id(self))


def _card_to_view(card: Card) -> CardView:
    return CardView(card.name, card.cost, card.is_active)


def view_hand(context: GameContext) -> list[CardView]:
    return [_card_to_view(card) for card in context.hand]


def view_draw_pile(context: GameContext) -> set[CardView]:
    return {_card_to_view(card) for card in context.draw_pile}


def view_discard_pile(context: GameContext) -> set[CardView]:
    return {_card_to_view(card) for card in context.discard_pile}
