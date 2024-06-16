from src.game.ecs.components.creatures import CreatureComponent
from src.game.ecs.components.creatures import CreatureHasModifiersComponent
from src.game.ecs.components.creatures import IsEndingTurnComponent
from src.game.ecs.components.creatures import ModifierStacksDurationComponent
from src.game.ecs.components.creatures import TurnEndComponent
from src.game.ecs.components.effects import EffectModifierDeltaComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: split into two systems, one for character and one for monster
class TurnEndingSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            creature_entity_id, _ = next(
                manager.get_component(CreatureComponent, IsEndingTurnComponent)
            )

        except StopIteration:
            return

        # Common effects
        creature_has_modifiers_component = manager.get_component_for_entity(
            creature_entity_id, CreatureHasModifiersComponent
        )
        if creature_has_modifiers_component is not None:
            for modifier_entity_id in creature_has_modifiers_component.modifier_entity_ids:
                # Tag modifier w/ TurnEnd
                manager.add_component(modifier_entity_id, TurnEndComponent())

            add_effect_to_bot(
                manager,
                manager.create_entity(
                    EffectModifierDeltaComponent(-1),
                    EffectQueryComponentsComponent(
                        [TurnEndComponent, ModifierStacksDurationComponent]
                    ),
                ),
            )

        # Untag & retag
        manager.destroy_component(IsEndingTurnComponent)
        manager.add_component(creature_entity_id, TurnEndComponent())
