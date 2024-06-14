from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import MonsterComponent
from src.game.ecs.components.creatures import MonsterCurrentMoveComponent
from src.game.ecs.components.creatures import MonsterHasMovesComponent
from src.game.ecs.components.creatures import MonsterMoveHasEffectsComponent
from src.game.ecs.components.creatures import TurnStartComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectSetBlockToZero
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: split into two systems, one for character and one for monster
# TODO: think about where to destroy IsTurnComponent
class TurnStartSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            creature_entity_id, _ = next(manager.get_component(TurnStartComponent))

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
                EffectSelectionTypeComponent(EffectSelectionType.NONE),
            ),
        )

        # Character-only effects
        if manager.get_component_for_entity(creature_entity_id, CharacterComponent) is not None:
            add_effect_to_bot(manager, manager.create_entity(EffectRefillEnergy()))
            add_effect_to_bot(manager, manager.create_entity(EffectDrawCardComponent(5)))

        # Monster-only effects
        elif manager.get_component_for_entity(creature_entity_id, MonsterComponent) is not None:
            # Get the monster's current move
            for move_entity_id in manager.get_component_for_entity(
                creature_entity_id, MonsterHasMovesComponent
            ).move_entity_ids:
                if (
                    manager.get_component_for_entity(move_entity_id, MonsterCurrentMoveComponent)
                    is not None
                ):
                    # Tag the move's effects to be dispatched
                    for effect_entity_id in manager.get_component_for_entity(
                        move_entity_id, MonsterMoveHasEffectsComponent
                    ).effect_entity_ids:
                        add_effect_to_bot(manager, effect_entity_id)

                    break

        # Untag creature & tag it w/ IsTurnComponent
        manager.remove_component(creature_entity_id, TurnStartComponent)
        manager.add_component(creature_entity_id, IsTurnComponent())
