from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterReadyToEndTurnComponent
from src.game.ecs.components.creatures import TurnStartComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class EndMonsterTurnSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            monster_entity_id, (monster_component, _) = next(
                manager.get_components(MonsterComponent, MonsterReadyToEndTurnComponent)
            )

        except StopIteration:
            return

        # Check if there's effects left to be processed
        if len(list(manager.get_component(EffectIsQueuedComponent))) > 0:
            return

        # If not, end the monster's turn
        manager.remove_component(monster_entity_id, MonsterReadyToEndTurnComponent)
        manager.remove_component(monster_entity_id, IsTurnComponent)

        # Pass IsTurn to next monster
        monster_is_turn_position = monster_component.position
        for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
            if monster_component.position == monster_is_turn_position + 1:
                # It's the next monster
                manager.add_component(monster_entity_id, IsTurnComponent())

                return

        # It's the player's turn
        char_entity_id, _ = next(manager.get_component(CharacterComponent))
        manager.add_component(char_entity_id, TurnStartComponent())
