from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import IsTurnComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterMoveComponent
from src.game.ecs.components.actors import MonsterMoveIsQueuedComponent
from src.game.ecs.components.actors import MonsterMoveParentComponent
from src.game.ecs.components.actors import TurnStartComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.components.effects import EffectSetBlockToZero
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: split into two systems, one for character and one for monster
# TODO: think about where to destroy IsTurnComponent
class TurnStartSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            actor_entity_id, _ = next(manager.get_component(TurnStartComponent))

        except StopIteration:
            return

        # Common effects
        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectSetBlockToZero(),
                EffectQueryComponentsComponent(
                    [IsTurnComponent],
                ),
            ),
        )

        # Character-only effects
        if manager.get_component_for_entity(actor_entity_id, CharacterComponent) is not None:
            add_effect_to_bot(manager, manager.create_entity(EffectRefillEnergy()))
            add_effect_to_bot(manager, manager.create_entity(EffectDrawCardComponent(5)))

        # Monster-only effects
        elif manager.get_component_for_entity(actor_entity_id, MonsterComponent) is not None:
            # Get the monster's current move
            for move_entity_id, (monster_move_parent_component, _) in manager.get_components(
                MonsterMoveParentComponent, MonsterMoveComponent
            ):
                if monster_move_parent_component.entity_id == actor_entity_id:
                    manager.add_component(move_entity_id, MonsterMoveIsQueuedComponent())

        # Untag actor & tag it w/ IsTurnComponent
        manager.remove_component(actor_entity_id, TurnStartComponent)
        manager.add_component(actor_entity_id, IsTurnComponent())
