from src.game.combat import input as input_
from src.game.combat.action import ActionType
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.cards import CardRequiresTargetComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
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

        if action_type == ActionType.SELECT_CARD:
            # TODO: check if there's enough energy here
            manager.add_component(action_target_entity_id, IsSelectedComponent())

            if (
                manager.get_component_for_entity(
                    action_target_entity_id, CardRequiresTargetComponent
                )
                is not None
            ):
                for monster_entity_id, _ in manager.get_component(MonsterComponent):
                    manager.add_component(monster_entity_id, CanBeSelectedComponent())

        # Select monster
        if action_type == ActionType.SELECT_MONSTER:
            card_is_selected_entity_id, _ = next(
                manager.get_components(CardInHandComponent, IsSelectedComponent)
            )
            manager.remove_component(card_is_selected_entity_id, IsSelectedComponent)
            manager.add_component(card_is_selected_entity_id, CardIsPlayedComponent())

            # Select
            manager.add_component(action_target_entity_id, IsSelectedComponent())

        # Confirm
        if action_type == ActionType.CONFIRM:
            card_is_selected_entity_id, _ = next(
                manager.get_components(CardInHandComponent, IsSelectedComponent)
            )
            manager.remove_component(card_is_selected_entity_id, IsSelectedComponent)
            manager.add_component(card_is_selected_entity_id, CardIsPlayedComponent())

        if action_type == ActionType.END_TURN:
            # Disable input
            manager.destroy_component(CanBeSelectedComponent)

            # Begin monsters' turn
            for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
                if monster_component.position == 0:
                    manager.add_component(monster_entity_id, IsTurnComponent())
