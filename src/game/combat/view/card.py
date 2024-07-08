from dataclasses import dataclass

from src.game.combat.entities import Entities


@dataclass
class CardView:
    entity_id: int
    name: str
    cost: int
    is_selectable: bool
    is_active: bool

    def __hash__(self) -> int:
        return hash(id(self))


def _card_to_view(entities: Entities, card_entity_id: int) -> CardView:
    card = entities.get_entity(card_entity_id)
    energy = entities.get_entity(entities.energy_id)

    return CardView(
        card_entity_id,
        card.name,
        card.cost,
        True if card.cost <= energy.current and entities.card_active_id is None else False,
        True if card_entity_id == entities.card_active_id else False,
    )


def view_draw_pile(entities: Entities) -> set[CardView]:
    return {
        _card_to_view(entities, card_entity_id)
        for card_entity_id in entities.card_in_draw_pile_ids
    }


def view_hand(entities: Entities) -> list[CardView]:
    return [
        _card_to_view(entities, card_entity_id) for card_entity_id in entities.card_in_hand_ids
    ]


def view_discard_pile(entities: Entities) -> set[CardView]:
    return {
        _card_to_view(entities, card_entity_id)
        for card_entity_id in entities.card_in_discard_pile_ids
    }
