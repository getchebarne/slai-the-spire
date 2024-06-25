from src.agents.base import BaseAgent
from src.game.combat.action import Action
from src.game.combat.action import ActionType
from src.game.combat.drawer import drawer
from src.game.combat.view import CombatView
from src.game.combat.view import get_combat_view
from src.game.ecs.components.action import ActionComponent
from src.game.ecs.components.action import ActionConfirmComponent
from src.game.ecs.components.action import ActionEndTurnComponent
from src.game.ecs.components.action import ActionSelectComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import TurnStartComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.common import IsSelectedComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectShuffleDeckIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.all import ALL_SYSTEMS


class CombatEngine:
    def _agent_should_make_action(self, manager: ECSManager) -> bool:
        # If it's not the character's turn, return False
        if len(list(manager.get_components(CharacterComponent, IsTurnComponent))) == 0:
            return False

        # If there's queued effects, return False
        if len(list(manager.get_component(EffectIsQueuedComponent))) > 0:
            return False

        return True

    def _get_action(self, agent: BaseAgent, view: CombatView) -> Action:
        return agent.select_action(view)

    def _is_game_over(self, manager: ECSManager) -> bool:
        # TODO: assume there's only one character
        return all(
            [
                character_health_component.current <= 0
                for _, (_, character_health_component) in manager.get_components(
                    CharacterComponent, HealthComponent
                )
            ]
        ) or all(
            [
                monster_health_component.current <= 0
                for _, (_, monster_health_component) in manager.get_components(
                    MonsterComponent, HealthComponent
                )
            ]
        )

    # TODO: this will have to be a system to activate on combat start effects, e.g., relics
    def _combat_start(self, manager: ECSManager) -> None:
        # Queue an effect to shuffle the deck into the draw pile
        manager.create_entity(EffectShuffleDeckIntoDrawPileComponent(), EffectIsQueuedComponent(0))

        # Start the character's turn
        character_entity_id, _ = list(manager.get_component(CharacterComponent))[0]
        manager.add_component(character_entity_id, TurnStartComponent())

    def _handle_agent_action(self, manager: ECSManager, action: Action) -> None:
        # Unpack
        action_type = action.type
        action_target_entity_id = action.target_entity_id

        # Check the selected entity is valid
        can_be_selected_entity_ids = [
            can_be_selected_entity_id
            for can_be_selected_entity_id, _ in manager.get_component(CanBeSelectedComponent)
        ]
        if (
            action_target_entity_id is not None
            and action_target_entity_id not in can_be_selected_entity_ids
        ):
            raise ValueError(f"Entity {action_target_entity_id} cannot be selected")

        # Create action entitiy
        action_entity_id = manager.create_entity(ActionComponent())

        # Select
        if action_type == ActionType.SELECT:
            manager.add_component(action_entity_id, ActionSelectComponent())
            manager.add_component(action_target_entity_id, IsSelectedComponent())

            return

        # Confirm
        if action_type == ActionType.CONFIRM:
            manager.add_component(action_entity_id, ActionConfirmComponent())

            return

        # End turn
        if action_type == ActionType.END_TURN:
            manager.add_component(action_entity_id, ActionEndTurnComponent())

    def _clear_action(self, manager: ECSManager) -> None:
        if query_result := list(manager.get_component(ActionComponent)):
            action_entity_id, _ = query_result[0]
            manager.destroy_entity(action_entity_id)
            manager.destroy_component(IsSelectedComponent)

    def run(self, manager: ECSManager, agent: BaseAgent) -> None:
        # Start combat
        self._combat_start(manager)

        while not self._is_game_over(manager):
            view = get_combat_view(manager)
            drawer(view)

            # Get action from agent
            self._clear_action(manager)
            if self._agent_should_make_action(manager):
                action = self._get_action(agent, view)
                self._handle_agent_action(manager, action)

            # Run systems
            for system in ALL_SYSTEMS:
                system.process(manager)
