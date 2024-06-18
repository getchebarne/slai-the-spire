from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import MonsterMoveDummyAttackComponent
from src.game.ecs.components.actors import MonsterMoveIsQueuedComponent
from src.game.ecs.components.actors import MonsterMoveParentComponent
from src.game.ecs.components.effects import EffectDealDamageComponent
from src.game.ecs.components.effects import EffectParentComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


DAMAGE = 5


# TODO: make effect parent be move instead of actor
class MoveDummyAttackSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(
            manager.get_components(
                MonsterMoveDummyAttackComponent,
                MonsterMoveIsQueuedComponent,
                MonsterMoveParentComponent,
            )
        )
        if query_result:
            move_entity_id, (_, _, monster_move_parent_component) = query_result[0]
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectDealDamageComponent(DAMAGE),
                    EffectQueryComponentsComponent([CharacterComponent]),
                    EffectParentComponent(monster_move_parent_component.entity_id),
                ),
            )

            manager.remove_component(move_entity_id, MonsterMoveIsQueuedComponent)
