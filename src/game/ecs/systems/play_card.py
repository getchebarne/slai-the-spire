from src.game.ecs.components.cards import CardCostComponent
from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.cards import CardIsPlayedSingletonComponent
from src.game.ecs.components.cards import CardLastPlayedComponent
from src.game.ecs.components.effects import EffectDiscardCardComponent
from src.game.ecs.components.effects import EffectQueryComponentsComponent
from src.game.ecs.components.energy import EnergyComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


class PlayCardSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        if not (query_result := list(manager.get_component(CardIsPlayedSingletonComponent))):
            return

        card_entity_id, _ = query_result[0]

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
                EffectQueryComponentsComponent([CardInHandComponent, CardLastPlayedComponent]),
            ),
        )

        # Untag the card
        manager.destroy_component(CardLastPlayedComponent)
        manager.add_component(card_entity_id, CardLastPlayedComponent())
