from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import ModifierStacksDurationComponent
from src.game.ecs.components.actors import MonsterComponent
from src.game.ecs.components.actors import MonsterPendingMoveUpdateComponent
from src.game.ecs.components.actors import NextTurnComponent
from src.game.ecs.components.actors import TurnComponent
from src.game.ecs.components.actors import TurnEndComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectIsTargetedSingletonComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.effects import EffectTurnEndComponent
from src.game.ecs.components.effects import EffectTurnStartComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: use different component for modifiers' turn end?
class TurnEndSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        if not list(
            manager.get_components(EffectTurnEndComponent, EffectIsTargetedSingletonComponent)
        ):
            return

        # Get current turn actor
        actor_entity_id, _ = list(manager.get_component(TurnComponent))[0]
        manager.destroy_component(TurnComponent)
        manager.add_component(actor_entity_id, TurnEndComponent())

        # Create effect to decrease actor's modifier's stacks
        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectModifierDeltaComponent(-1),
                EffectQueryComponentsComponent(
                    [ModifierStacksDurationComponent, TurnEndComponent]
                ),
                EffectSelectionTypeComponent(EffectSelectionType.ALL),
            ),
        )

        if manager.get_component_for_entity(actor_entity_id, CharacterComponent) is not None:
            # Add effect to discard the hand
            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectDiscardCardComponent(),
                    EffectQueryComponentsComponent([CardInHandComponent]),
                    EffectSelectionTypeComponent(EffectSelectionType.ALL),
                ),
            )
            for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
                if monster_component.position == 0:
                    next_actor_entity_id = monster_entity_id
                    break

        elif (
            monster_component := manager.get_component_for_entity(
                actor_entity_id, MonsterComponent
            )
        ) is not None:
            # Tag monster as pending move update
            manager.add_component(actor_entity_id, MonsterPendingMoveUpdateComponent())

            # Find next monster
            monster_is_turn_position = monster_component.position
            next_actor_entity_id = None
            for monster_entity_id, monster_component in manager.get_component(MonsterComponent):
                if monster_component.position == monster_is_turn_position + 1:
                    next_actor_entity_id = monster_entity_id
                    break

            if next_actor_entity_id is None:
                # The next turn is the player's
                next_actor_entity_id, _ = list(manager.get_component(CharacterComponent))[0]

        # Tag next actor to take a turn
        manager.destroy_component(NextTurnComponent)
        manager.add_component(next_actor_entity_id, NextTurnComponent())

        # Create an effect at the end to start the next actor's turn
        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectTurnStartComponent(), EffectQueryComponentsComponent([NextTurnComponent])
            ),
        )
