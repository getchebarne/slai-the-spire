from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import effect_queue_is_empty
from src.game.ecs.utils import trigger_actor_turn_end


# TODO: maybe consolidate into single EndTurn system?
class ProcessMonsterTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, (monster_component, _) = next(
                manager.get_components(MonsterComponent, IsTurnComponent)
            )

        except StopIteration:
            return

        # Check if there's effects in the queue
        # TODO improve comment
        if not effect_queue_is_empty(manager):
            return

        # If not, end the monster's turn
        manager.remove_component(monster_entity_id, IsTurnComponent)
        trigger_actor_turn_end(manager, monster_entity_id)
