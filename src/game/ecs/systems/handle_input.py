from src.game.combat import input as input_
from src.game.combat.action import ActionType
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.cards import CardTargetComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import TurnEndComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


def _promote_card_from_selected_to_played(manager: ECSManager) -> None:
    # Get currently selected card
    card_entity_id, _ = next(manager.get_components(CardInHandComponent, IsSelectedComponent))

    # Untag card as selected & tag it as played
    manager.remove_component(card_entity_id, IsSelectedComponent)
    manager.add_component(card_entity_id, CardIsPlayedComponent())


def _handle_select_card(manager: ECSManager, action_target_entity_id: int) -> None:
    # Deselect current selected entities
    manager.destroy_component(IsSelectedComponent)

    # Tag card as selected
    manager.add_component(action_target_entity_id, IsSelectedComponent())


def _handle_confirm(manager: ECSManager) -> None:
    _promote_card_from_selected_to_played(manager)


def _handle_select_monster(manager: ECSManager, action_target_entity_id: int) -> None:
    _promote_card_from_selected_to_played(manager)

    # Tag monster as card's target
    manager.add_component(action_target_entity_id, CardTargetComponent())


def _handle_end_turn(manager: ECSManager) -> None:
    # Get character
    character_entity_id, _ = next(manager.get_component(CharacterComponent))

    # Finish its turn and trigger its turn end. TODO: improve comment
    manager.remove_component(character_entity_id, IsTurnComponent)
    manager.add_component(character_entity_id, TurnEndComponent())


class HandleInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        if input_.action is None:
            return

        action_type = input_.action.type
        action_target_entity_id = input_.action.target_entity_id

        can_be_selected_entity_ids = [
            can_be_selected_entity_id
            for can_be_selected_entity_id, _ in manager.get_component(CanBeSelectedComponent)
        ]
        if (
            action_target_entity_id is not None
            and action_target_entity_id not in can_be_selected_entity_ids
        ):
            raise ValueError(f"Entity {action_target_entity_id} cannot be selected")

        # Select card
        if action_type == ActionType.SELECT_CARD:
            _handle_select_card(manager, action_target_entity_id)

        # Confirm
        if action_type == ActionType.CONFIRM:
            _handle_confirm(manager)

        # Select monster
        if action_type == ActionType.SELECT_MONSTER:
            _handle_select_monster(manager, action_target_entity_id)

        # End turn
        if action_type == ActionType.END_TURN:
            _handle_end_turn(manager)
