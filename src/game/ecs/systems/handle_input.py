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


class HandleInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # TODO: this doesn't go here
        manager.destroy_component(CardIsPlayedComponent)

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

        if action_type == ActionType.SELECT_CARD:
            # TODO: check if there's enough energy here
            manager.add_component(action_target_entity_id, IsSelectedComponent())

        # Select monster
        if action_type == ActionType.SELECT_MONSTER:
            card_is_selected_entity_id, _ = next(
                manager.get_components(CardInHandComponent, IsSelectedComponent)
            )
            manager.remove_component(card_is_selected_entity_id, IsSelectedComponent)
            manager.add_component(card_is_selected_entity_id, CardIsPlayedComponent())

            # Select
            manager.add_component(action_target_entity_id, CardTargetComponent())

        # Confirm. TODO: maybe remove confirm
        if action_type == ActionType.CONFIRM:
            card_is_selected_entity_id, _ = next(
                manager.get_components(CardInHandComponent, IsSelectedComponent)
            )
            manager.remove_component(card_is_selected_entity_id, IsSelectedComponent)
            manager.add_component(card_is_selected_entity_id, CardIsPlayedComponent())

        if action_type == ActionType.END_TURN:
            character_entity_id, _ = next(manager.get_component(CharacterComponent))
            manager.remove_component(character_entity_id, IsTurnComponent)
            manager.add_component(character_entity_id, TurnEndComponent())
