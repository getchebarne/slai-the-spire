from src.agents.base import BaseAgent
from src.game.combat import input as input_
from src.game.combat.view import combat_view
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import DrawCardEffectComponent
from src.game.ecs.components.effects import EffectIsTargetedComponent
from src.game.ecs.components.effects import ShuffleDeckIntoDrawPileEffectComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.all import ALL_SYSTEMS


class CombatEngine:

    def _character_turn(self, manager: ECSManager, agent: BaseAgent) -> None:
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

    def _init(self, manager: ECSManager) -> None:
        manager.create_entity(
            ShuffleDeckIntoDrawPileEffectComponent(), EffectIsTargetedComponent()
        )
        manager.create_entity(DrawCardEffectComponent(5), EffectIsTargetedComponent())

    def run(self, manager: ECSManager, agent: BaseAgent) -> None:
        self._init(manager)

        while not self._is_game_over(manager):
            view = combat_view(manager)
            print(view.monsters)
            print(view.character)
            print(view.hand)
            print(view.discard_pile)
            print(view.energy)
            print("######################")
            # Run systems
            for system in ALL_SYSTEMS:
                system.process(manager)

            self._character_turn(manager, agent)
