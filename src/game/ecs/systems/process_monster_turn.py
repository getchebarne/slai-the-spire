from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.effects import DrawCardEffectComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.components.effects import RefillEnergyEffect
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

        # Execute move
        # print("executing move")
        # # Update move
        # print("updating move")

        # Untag
        manager.remove_component(is_turn_monster_entity_id, IsTurnComponent)

        # Pass IsTurn to next monster
        monster_is_turn_position = monster_component.position
        for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
            if monster_component.position == monster_is_turn_position + 1:
                # It's the next monster
                manager.add_component(monster_entity_id, IsTurnComponent())

                return

        # It's the player's turn. TODO: will probably have to create a "CharacterTurnStartEffect"
        # or sth like that
        char_entity_id, _ = next(manager.get_component(CharacterComponent))
        manager.add_component(char_entity_id, IsTurnComponent())
        manager.create_entity(RefillEnergyEffect(), EffectToBeDispatchedComponent(0))
        manager.create_entity(DrawCardEffectComponent(5), EffectToBeDispatchedComponent(1))
