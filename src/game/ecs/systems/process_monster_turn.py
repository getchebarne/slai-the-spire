from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterMoveComponent
from src.game.ecs.components.creatures import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.creatures import MonsterReadyToEndTurnComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class ProcessMonsterTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        # Get monster's entity id
        try:
            is_turn_monster_entity_id, (monster_component, _) = next(
                manager.get_components(MonsterComponent, IsTurnComponent)
            )

        except StopIteration:
            return

        # Get the monster's current move
        monster_move_component = manager.get_component_for_entity(
            is_turn_monster_entity_id, MonsterMoveComponent
        )

        # Tag the move's effects to be dispatched
        for priority, effect_entity_id in enumerate(monster_move_component.effect_entity_ids):
            manager.add_component(
                effect_entity_id,
                EffectIsQueuedComponent(priority=priority + 1),  # TODO: revisit
            )

        # Tag monster as pending move update & ready to end its turn
        manager.add_component(is_turn_monster_entity_id, MonsterPendingMoveUpdateComponent())
        manager.add_component(is_turn_monster_entity_id, MonsterReadyToEndTurnComponent())
