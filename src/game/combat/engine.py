from src.agents.base import BaseAgent
from src.game.combat import input as input_
from src.game.combat.action import Action
from src.game.combat.drawer import drawer
from src.game.combat.view import CombatView
from src.game.combat.view import get_combat_view
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import HealthComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import TurnStartComponent
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

    def run(self, manager: ECSManager, agent: BaseAgent) -> None:
        # Start combat
        self._combat_start(manager)

        while not self._is_game_over(manager):
            view = get_combat_view(manager)
            drawer(view)

            # Get action from agent
            input_.action = None
            if self._agent_should_make_action(manager):
                input_.action = self._get_action(agent, view)

            # Run systems
            for system in ALL_SYSTEMS:
                system.process(manager)
