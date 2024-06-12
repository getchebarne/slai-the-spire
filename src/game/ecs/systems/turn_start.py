from src.game.ecs.components.creatures import CharacterComponent
from src.game.ecs.components.creatures import IsTurnComponent
from src.game.ecs.components.creatures import TurnStartComponent
from src.game.ecs.components.effects import EffectDrawCardComponent
from src.game.ecs.components.effects import EffectRefillEnergy
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.utils import add_effect_to_bot


# TODO: add monster logic
class TurnStartSystem(BaseSystem):
    def process(self, manager: ECSManager) -> None:
        try:
            creature_entity_id, _ = next(manager.get_component(TurnStartComponent))

        except StopIteration:
            return

        # Character-only effects
        if manager.get_component_for_entity(creature_entity_id, CharacterComponent) is not None:
            add_effect_to_bot(manager, EffectDrawCardComponent(5))
            add_effect_to_bot(manager, EffectRefillEnergy())

        # Untag creature & tag it w/ IsTurnComponent
        manager.remove_component(creature_entity_id, TurnStartComponent)
        manager.add_component(creature_entity_id, IsTurnComponent())
