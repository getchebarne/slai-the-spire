from src.game.ecs.components.cards import CardInHandComponent
from src.game.ecs.components.common import CanBeSelectedComponent
from src.game.ecs.components.effects import EffectToBeDispatchedComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


# TODO: improve this it's awful
class EnableInputSystem(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        if len(list(manager.get_component(EffectToBeDispatchedComponent))) > 0:
            manager.destroy_component(CanBeSelectedComponent)
            return

        if len(list(manager.get_component(CanBeSelectedComponent))) > 0:
            return

        for card_in_hand_entity_id, _ in manager.get_component(CardInHandComponent):
            manager.add_component(card_in_hand_entity_id, CanBeSelectedComponent())
