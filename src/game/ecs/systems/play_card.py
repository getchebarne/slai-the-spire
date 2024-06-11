from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectIsQueuedComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem


class PlayCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            card_entity_id, _ = next(manager.get_component(CardIsPlayedComponent))

        except StopIteration:
            return

        # Get the card's cost
        card_cost = manager.get_component_for_entity(card_entity_id, CardCostComponent).value

        # Check if there's enough energy to play the card
        energy_component = list(manager.get_component(EnergyComponent))[0][1]
        if energy_component.current < card_cost:
            raise ValueError("Not enough energy to play card")

        # Decrease the energy. TODO: this should be an effect
        energy_component.current -= card_cost

        # Tag the card's effects to be dispatched
        for priority, effect_entity_id in enumerate(
            manager.get_component_for_entity(
                card_entity_id, CardHasEffectsComponent
            ).effect_entity_ids
        ):
            manager.add_component(
                effect_entity_id,
                EffectIsQueuedComponent(priority=priority + 1),  # TODO: revisit
            )

        # Create effect to discard the played card
        manager.create_entity(
            EffectDiscardCardComponent(),
            EffectIsQueuedComponent(priority=0),
            EffectQueryComponentsComponent([CardIsPlayedComponent]),
            EffectSelectionTypeComponent(EffectSelectionType.NONE),
        )

        return
