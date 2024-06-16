import random

from src.game.ecs.components.actors import DummyAIComponent
from src.game.ecs.components.actors import MonsterCurrentMoveComponent
from src.game.ecs.components.actors import MonsterHasMovesComponent
from src.game.ecs.components.actors import MonsterMoveComponent
from src.game.ecs.components.actors import MonsterPendingMoveUpdateComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


# TODO: improve how moves are defined
class AIDummySystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, (monster_has_moves_component, _, _) = next(
                manager.get_components(
                    MonsterHasMovesComponent, DummyAIComponent, MonsterPendingMoveUpdateComponent
                )
            )

        except StopIteration:
            return

        # Get current move
        current_move_name = None
        move_name_entity_ids = {}
        for move_entity_id in monster_has_moves_component.move_entity_ids:
            move_name = manager.get_component_for_entity(move_entity_id, MonsterMoveComponent).name
            move_name_entity_ids[move_name] = move_entity_id

            if (
                manager.get_component_for_entity(move_entity_id, MonsterCurrentMoveComponent)
                is not None
            ):
                current_move_name = move_name

        if current_move_name is None:
            manager.add_component(
                random.choice(monster_has_moves_component.move_entity_ids),
                MonsterCurrentMoveComponent(),
            )

        else:
            manager.remove_component(
                move_name_entity_ids[current_move_name], MonsterCurrentMoveComponent
            )
            if current_move_name == "Attack":
                manager.add_component(
                    move_name_entity_ids["Defend"], MonsterCurrentMoveComponent()
                )

            elif current_move_name == "Defend":
                manager.add_component(
                    move_name_entity_ids["Attack"], MonsterCurrentMoveComponent()
                )

            else:
                raise ValueError(f"Invalid move name: {current_move_name}")

        # Untag
        manager.remove_component(monster_entity_id, MonsterPendingMoveUpdateComponent)
