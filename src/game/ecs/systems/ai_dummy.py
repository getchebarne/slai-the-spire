import random

from src.game.ecs.components.actors import DummyAIComponent
from src.game.ecs.components.actors import MonsterMoveDummyAttackComponent
from src.game.ecs.components.actors import MonsterMoveDummyDefendComponent
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

        manager.remove_component(monster_entity_id, MonsterPendingMoveUpdateComponent)

        # Get current move
        if (
            manager.get_component_for_entity(monster_entity_id, MonsterMoveDummyAttackComponent)
            is not None
        ):
            manager.remove_component(monster_entity_id, MonsterMoveDummyAttackComponent)
            manager.add_component(monster_entity_id, MonsterMoveDummyDefendComponent())

            return

        if (
            manager.get_component_for_entity(monster_entity_id, MonsterMoveDummyDefendComponent)
            is not None
        ):
            manager.remove_component(monster_entity_id, MonsterMoveDummyDefendComponent)
            manager.add_component(monster_entity_id, MonsterMoveDummyAttackComponent())

            return

        manager.add_component(
            monster_entity_id,
            random.choice([MonsterMoveDummyAttackComponent, MonsterMoveDummyDefendComponent])(),
        )
