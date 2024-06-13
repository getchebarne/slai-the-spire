from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardHasEffectsComponent
from src.game.ecs.components.cards import CardIsPlayedComponent
from src.game.ecs.components.cards import CardLastPlayedComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.effects import EffectSelectionType
from src.game.ecs.components.effects import EffectSelectionTypeComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


class PlayCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            card_entity_id, _ = next(manager.get_component(CardIsPlayedComponent))

        except StopIteration:
            return

        # Decrease the energy
        # TODO: this should be an effect
        card_cost = manager.get_component_for_entity(card_entity_id, CardCostComponent).value
        _, energy_component = next(manager.get_component(EnergyComponent))
        energy_component.current -= card_cost

        # Create effect to discard the played card
        add_effect_to_bot(
            manager,
            manager.create_entity(
                EffectDiscardCardComponent(),
                EffectQueryComponentsComponent([CardLastPlayedComponent]),
                EffectSelectionTypeComponent(EffectSelectionType.NONE),
            ),
        )

        # Tag the card's effects to be dispatched
        for effect_entity_id in manager.get_component_for_entity(
            card_entity_id, CardHasEffectsComponent
        ).effect_entity_ids:
            add_effect_to_bot(manager, effect_entity_id)

        # Untag & retag the card
        manager.destroy_component(CardIsPlayedComponent)
        manager.destroy_component(CardLastPlayedComponent)
        manager.add_component(card_entity_id, CardLastPlayedComponent())
