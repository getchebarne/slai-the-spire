from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import TurnEndComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: maybe consolidate into single EndTurn system?
class ProcessMonsterTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, (monster_component, _) = next(
                manager.get_components(MonsterComponent, IsTurnComponent)
            )

        except StopIteration:
            return

        # Check if there's effects left to be processed
        if len(list(manager.get_component(EffectIsQueuedComponent))) > 0:
            return

        # If not, end the monster's turn
        manager.remove_component(monster_entity_id, IsTurnComponent)
        manager.add_component(monster_entity_id, TurnEndComponent())
