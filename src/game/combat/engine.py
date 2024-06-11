from src.agents.base import BaseAgent
from src.game.combat import input as input_
from src.game.combat.view import combat_view
from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import HealthComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectShuffleDeckIntoDrawPileComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.all import ALL_SYSTEMS


class CombatEngine:

    def _character_turn(self, manager: ECSManager, agent: BaseAgent) -> None:
        view = combat_view(manager)
        input_.action = agent.select_action(view)
        print(input_.action)

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
        manager.create_entity(EffectShuffleDeckIntoDrawPileComponent(), EffectIsQueuedComponent(0))
        manager.create_entity(EffectDrawCardComponent(5), EffectIsQueuedComponent(1))
        manager.add_component(
            next(manager.get_component(CharacterComponent))[0], IsTurnComponent()
        )

    def run(self, manager: ECSManager, agent: BaseAgent) -> None:
        self._init(manager)

        i = 0
        while not self._is_game_over(manager):
            view = combat_view(manager)
            print(view.monsters)
            print(view.character)
            print(view.hand)
            print(view.discard_pile)
            print(view.energy)

            # Run systems
            self._character_turn(manager, agent)
            for system in ALL_SYSTEMS:
                system.process(manager)

            print("######################")
            i += 1
            # if i > 3:
            #     exit()
