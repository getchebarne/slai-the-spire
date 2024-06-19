from src.game.ecs.components.actors import DummyAIComponent
from src.game.ecs.components.actors import MonsterIntentBlockComponent
from src.game.ecs.components.actors import MonsterIntentDamageComponent
from src.game.ecs.components.actors import MonsterIntentTimesComponent
from src.game.ecs.components.actors import MonsterMoveDummyAttackComponent
from src.game.ecs.components.actors import MonsterMoveDummyDefendComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: define in a database so that there's no duplicate code between here
# and the move systems
DAMAGE = 5
TIMES = 1
BLOCK = 5


class IntentDummySystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        for monster_entity_id, _ in manager.get_component(DummyAIComponent):
            # Get current move
            if (
                manager.get_component_for_entity(
                    monster_entity_id, MonsterMoveDummyAttackComponent
                )
                is not None
            ):
                if (
                    manager.get_component_for_entity(
                        monster_entity_id, MonsterIntentBlockComponent
                    )
                    is not None
                ):
                    manager.remove_component(monster_entity_id, MonsterIntentBlockComponent)

                manager.add_component(monster_entity_id, MonsterIntentDamageComponent(DAMAGE))
                manager.add_component(monster_entity_id, MonsterIntentTimesComponent(TIMES))

            elif (
                manager.get_component_for_entity(
                    monster_entity_id, MonsterMoveDummyDefendComponent
                )
                is not None
            ):
                if (
                    manager.get_component_for_entity(
                        monster_entity_id, MonsterIntentDamageComponent
                    )
                    is not None
                    and manager.get_component_for_entity(
                        monster_entity_id, MonsterIntentTimesComponent
                    )
                    is not None
                ):
                    manager.remove_component(monster_entity_id, MonsterIntentDamageComponent)
                    manager.remove_component(monster_entity_id, MonsterIntentTimesComponent)

                manager.add_component(monster_entity_id, MonsterIntentBlockComponent())
