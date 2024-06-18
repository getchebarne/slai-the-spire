import random

from src.game.ecs.components.actors import DummyAIComponent
from src.game.ecs.components.actors import MonsterMoveComponent
from src.game.ecs.components.actors import MonsterMoveDummyAttackComponent
from src.game.ecs.components.actors import MonsterMoveDummyDefendComponent
from src.game.ecs.components.actors import MonsterMoveParentComponent
from src.game.ecs.components.actors import MonsterPendingMoveUpdateComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class AIDummySystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, _ = next(
                manager.get_components(DummyAIComponent, MonsterPendingMoveUpdateComponent)
            )

        except StopIteration:
            return

        # Get current move
        found = False
        for move_entity_id, (monster_move_parent_component, _) in manager.get_components(
            MonsterMoveParentComponent, MonsterMoveComponent
        ):
            if monster_move_parent_component.entity_id == monster_entity_id:
                found = True
                if (
                    manager.get_component_for_entity(
                        move_entity_id, MonsterMoveDummyAttackComponent
                    )
                    is not None
                ):
                    manager.remove_component(move_entity_id, MonsterMoveDummyAttackComponent)
                    manager.add_component(move_entity_id, MonsterMoveDummyDefendComponent())
                elif (
                    manager.get_component_for_entity(
                        move_entity_id, MonsterMoveDummyDefendComponent
                    )
                    is not None
                ):
                    manager.remove_component(move_entity_id, MonsterMoveDummyDefendComponent)
                    manager.add_component(move_entity_id, MonsterMoveDummyAttackComponent())

        if not found:
            manager.create_entity(
                MonsterMoveComponent(),
                MonsterMoveParentComponent(monster_entity_id),
                random.choice(
                    [MonsterMoveDummyAttackComponent(), MonsterMoveDummyDefendComponent()]
                ),
            )
