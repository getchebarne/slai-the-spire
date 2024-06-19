from src.game.ecs.components.actors import ActorComponent
from src.game.ecs.components.actors import BeforeTurnEndComponent
from src.game.ecs.components.actors import CharacterComponent
from src.game.ecs.components.actors import ModifierParentComponent
from src.game.ecs.components.actors import ModifierStacksDurationComponent
from src.game.ecs.components.actors import TurnEndComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: use different component for modifiers' turn end?
class BeforeTurnEndSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        query_result = list(manager.get_components(ActorComponent, BeforeTurnEndComponent))
        if query_result:
            actor_entity_id, _ = query_result[0]

            # Tag actor's modifiers
            for modifier_entity_id, modifier_parent_component in manager.get_component(
                ModifierParentComponent
            ):
                if modifier_parent_component.actor_entity_id == actor_entity_id:
                    manager.add_component(modifier_entity_id, TurnEndComponent())

            # If the actor is the character, create an effect to discard the hand
            if manager.get_component_for_entity(actor_entity_id, CharacterComponent) is not None:
                add_effect_to_bot(
                    manager,
                    manager.create_entity(
                        EffectDiscardCardComponent(),
                        EffectQueryComponentsComponent([CardInHandComponent]),
                        EffectSelectionTypeComponent(EffectSelectionType.ALL),
                    ),
                )

            # Create effect to decrease modifier stacks
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
            # Promote actor
            manager.remove_component(actor_entity_id, BeforeTurnEndComponent)
            manager.add_component(actor_entity_id, TurnEndComponent())
