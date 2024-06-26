from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import TurnComponent
from src.game.ecs.components.effects import EffectTurnEndComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot
from src.game.ecs.utils import effect_queue_is_empty


# TODO: maybe consolidate into single EndTurn system?
class ProcessMonsterTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        if not (query_result := list(manager.get_components(MonsterComponent, TurnComponent))):
            return

        monster_entity_id, (monster_component, _) = query_result[0]

        # Check if there's effects in the queue
        if not effect_queue_is_empty(manager):
            return

        # If not, end the monster's turn
        add_effect_to_bot(manager, manager.create_entity(EffectTurnEndComponent()))
