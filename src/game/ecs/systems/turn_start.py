from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterMoveIsQueuedComponent
from src.game.ecs.components.actors import TurnComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectRefillEnergyComponent
from src.game.ecs.components.effects import EffectSetBlockToZero
from src.game.ecs.components.effects import EffectTargetComponent
from src.game.ecs.components.effects import EffectTurnStartComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


class TurnStartSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        if not list(
            manager.get_components(EffectTurnStartComponent, EffectIsTargetedSingletonComponent)
        ):
            return

        # Get current turn actor
        actor_entity_id, _ = list(manager.get_component(EffectTargetComponent))[0]
        manager.add_component(actor_entity_id, TurnComponent())

        # Find the id of the next actor
        if manager.get_component_for_entity(actor_entity_id, CharacterComponent) is not None:
            # Character-only effects
            add_effect_to_bot(manager, manager.create_entity(EffectRefillEnergyComponent()))
            add_effect_to_bot(manager, manager.create_entity(EffectDrawCardComponent(5)))

        elif manager.get_component_for_entity(actor_entity_id, MonsterComponent) is not None:
            # Queue monster's move
            manager.add_component(actor_entity_id, MonsterMoveIsQueuedComponent())

        # Common effects
        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectSetBlockToZero(), EffectQueryComponentsComponent([TurnComponent])
            ),
        )
