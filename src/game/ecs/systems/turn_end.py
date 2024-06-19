from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.actors import TurnEndComponent
from src.game.ecs.components.actors import TurnStartComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import effect_queue_is_empty


# TODO: split into two systems, one for character and one for monster
class TurnEndSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            actor_entity_id, _ = next(manager.get_components(ActorComponent, TurnEndComponent))

        except StopIteration:
            return

        if not effect_queue_is_empty(manager):
            return

        # TODO: reorder
        manager.destroy_component(TurnEndComponent)

        # If the actor is the character, start the monsters' turn
        if manager.get_component_for_entity(actor_entity_id, CharacterComponent) is not None:
            for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
                if monster_component.position == 0:
                    manager.add_component(monster_entity_id, TurnStartComponent())
                    break

        # Monster logic
        monster_component = manager.get_component_for_entity(actor_entity_id, MonsterComponent)
        if monster_component is not None:
            # Tag monster as pending move update
            manager.add_component(actor_entity_id, MonsterPendingMoveUpdateComponent())

            # Pass IsTurn to next monster
            monster_is_turn_position = monster_component.position
            for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
                if monster_component.position == monster_is_turn_position + 1:
                    # It's the next monster
                    manager.add_component(monster_entity_id, TurnStartComponent())

                    return

            # It's the player's turn
            char_entity_id, _ = next(manager.get_component(CharacterComponent))
            manager.add_component(char_entity_id, TurnStartComponent())
