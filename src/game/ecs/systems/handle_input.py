from src.game.combat import input as input_
from src.game.combat.action import ActionType
from src.game.ecs.components.actors import BeforeTurnEndComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


def _handle_select_entity(manager: ECSManager, action_target_entity_id: int) -> None:
    # Tag the entitiy as selected
    manager.add_component(action_target_entity_id, IsSelectedComponent())


def _handle_confirm(manager: ECSManager) -> None:
    # Get active card & tag it as played
    card_entity_id, _ = next(manager.get_components(CardIsActiveComponent))
    manager.add_component(card_entity_id, CardIsPlayedComponent())


def _handle_end_turn(manager: ECSManager) -> None:
    # Get character
    character_entity_id, _ = next(manager.get_component(CharacterComponent))

    # Finish its turn and trigger its turn end. TODO: improve comment
    manager.remove_component(character_entity_id, IsTurnComponent)
    manager.add_component(character_entity_id, BeforeTurnEndComponent())


class HandleInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # Clear previous selection
        manager.destroy_component(IsSelectedComponent)

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
        if action_type == ActionType.SELECT_ENTITY:
            _handle_select_entity(manager, action_target_entity_id)

        # Confirm
        if action_type == ActionType.CONFIRM:
            _handle_confirm(manager)

        # End turn
        if action_type == ActionType.END_TURN:
            _handle_end_turn(manager)
