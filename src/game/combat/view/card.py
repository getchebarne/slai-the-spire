from dataclasses import dataclass

from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import NameComponent
from src.game.ecs.manager import ECSManager


@dataclass
class CardView:
    entity_id: int
    name: str
    cost: int
    can_be_selected: bool
    is_active: bool


def get_card_view(entity_id: int, manager: ECSManager) -> CardView:
    name_component = manager.get_component_for_entity(entity_id, NameComponent)
    card_cost_component = manager.get_component_for_entity(entity_id, CardCostComponent)
    can_be_selected_component = manager.get_component_for_entity(entity_id, CanBeSelectedComponent)
    card_is_active_component = manager.get_component_for_entity(entity_id, CardIsActiveComponent)

    return CardView(
        entity_id,
        name_component.value,
        card_cost_component.value,
        False if can_be_selected_component is None else True,
        False if card_is_active_component is None else True,
    )


def get_hand_view(manager: ECSManager) -> list[CardView]:
    # Create a list of tuples containing CardView objects and their positions
    hand_view = [
        (get_card_view(entity_id, manager), card_in_hand_component.position)
        for entity_id, card_in_hand_component in manager.get_component(CardInHandComponent)
    ]

    # Sort the list of tuples by the position
    hand_view.sort(key=lambda x: x[1])

    # Extract the sorted hand list
    hand_view = [card_view for card_view, _ in hand_view]

    return hand_view
