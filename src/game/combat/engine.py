from src.agents.base import BaseAgent
from src.game.combat import input as input_
from src.game.combat.view import combat_view
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import TurnStartComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectShuffleDeckIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.all import ALL_SYSTEMS
from src.game.combat.drawer import draw_view


class CombatEngine:
    def _get_action(self, manager: ECSManager, agent: BaseAgent) -> None:
        view = combat_view(manager)
        input_.action = agent.select_action(view)

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
        character_entity_id, _ = next(manager.get_component(CharacterComponent))
        manager.add_component(character_entity_id, TurnStartComponent())

    def run(self, manager: ECSManager, agent: BaseAgent) -> None:
        # Start combat
        self._combat_start(manager)

        while not self._is_game_over(manager):
            view = combat_view(manager)
            draw_view(view)

            # Get action from agent
            self._get_action(manager, agent)

            # Run systems
            for system in ALL_SYSTEMS:
                system.process(manager)
