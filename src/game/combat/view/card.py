from dataclasses import dataclass

from src.game.combat.state import GameState


@dataclass
class CardView:
    entity_id: int
    name: str
    cost: int
    is_active: bool  # TODO: make int

    def __hash__(self) -> int:
        return hash(id(self))


def _card_to_view(state: GameState, card_entity_id: int) -> CardView:
    card = state.get_entity(card_entity_id)

    return CardView(
        card_entity_id,
        card.name,
        card.cost,
        True if card_entity_id == state.card_active_id else False,
    )


def view_draw_pile(state: GameState) -> set[CardView]:
    return {_card_to_view(state, card_entity_id) for card_entity_id in state.card_in_draw_pile_ids}


def view_hand(state: GameState) -> list[CardView]:
    return [_card_to_view(state, card_entity_id) for card_entity_id in state.card_in_hand_ids]


def view_discard_pile(state: GameState) -> set[CardView]:
    return {
        _card_to_view(state, card_entity_id) for card_entity_id in state.card_in_discard_pile_ids
    }
