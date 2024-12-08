from dataclasses import dataclass

from src.game.combat.entities import CardName
from src.game.combat.entities import Entities
from src.game.combat.entities import Effect


@dataclass
class CardView:
    entity_id: int
    name: CardName
    effects: list[Effect]  # TODO: implement EffectView
    cost: int
    is_active: bool

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CardView):
            return False

        return id(self) == id(other)


def _card_to_view(entities: Entities, card_entity_id: int) -> CardView:
    card = entities.get_entity(card_entity_id)

    return CardView(
        card_entity_id,
        card.name,
        card.effects,
        card.cost,
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
