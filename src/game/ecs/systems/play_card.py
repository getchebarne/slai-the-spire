from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardIsActiveComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class PlayCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        query_result = list(manager.get_component(CardIsActiveComponent))
        if len(query_result) == 0:
            return ProcessStatus.COMPLETE

        if len(query_result) > 1:
            raise ValueError("There can only be one active card at a time")

        card_entity_id, _ = query_result[0]

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
                effect_entity_id, EffectToBeDispatchedComponent(priority=priority)
            )

        # Untag the active card
        manager.remove_component(card_entity_id, CardIsActiveComponent)

        return ProcessStatus.COMPLETE
