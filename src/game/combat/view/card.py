from dataclasses import dataclass

from src.game.combat.state import Card
from src.game.combat.state import GameState


@dataclass
class CardView:
    name: str
    cost: int
    is_active: bool  # TODO: make int

    def __hash__(self) -> int:
        return hash(id(self))


def _card_to_view(state: GameState, card: Card) -> CardView:
    return CardView(
        card.name,
        card.cost,
        True if card is state.get_active_card() else False,  # TODO: revisit
    )


def view_hand(state: GameState) -> list[CardView]:
    return [_card_to_view(state, card) for card in state.get_hand()]


def view_draw_pile(state: GameState) -> set[CardView]:
    return {_card_to_view(state, card) for card in state.get_draw_pile()}


def view_discard_pile(state: GameState) -> set[CardView]:
    return {_card_to_view(state, card) for card in state.get_discard_pile()}
