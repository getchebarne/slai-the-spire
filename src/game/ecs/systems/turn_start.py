from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterMoveComponent
from src.game.ecs.components.creatures import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.creatures import MonsterReadyToEndTurnComponent
from src.game.ecs.components.creatures import TurnStartComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: add monster logic
class TurnStartSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            creature_entity_id, _ = next(manager.get_component(TurnStartComponent))

        except StopIteration:
            return

        # Character-only effects
        if manager.get_component_for_entity(creature_entity_id, CharacterComponent) is not None:
            add_effect_to_bot(manager, EffectDrawCardComponent(5))
            add_effect_to_bot(manager, EffectRefillEnergy())

        elif manager.get_component_for_entity(creature_entity_id, MonsterComponent) is not None:
            # Get the monster's current move
            monster_move_component = manager.get_component_for_entity(
                creature_entity_id, MonsterMoveComponent
            )

            # Tag the move's effects to be dispatched
            for priority, effect_entity_id in enumerate(monster_move_component.effect_entity_ids):
                manager.add_component(
                    effect_entity_id,
                    EffectIsQueuedComponent(priority=priority + 1),  # TODO: revisit
                )

            # Tag monster as pending move update & ready to end its turn
            manager.add_component(creature_entity_id, MonsterPendingMoveUpdateComponent())
            manager.add_component(creature_entity_id, MonsterReadyToEndTurnComponent())

        # Untag creature & tag it w/ IsTurnComponent
        manager.remove_component(creature_entity_id, TurnStartComponent)
        manager.add_component(creature_entity_id, IsTurnComponent())
