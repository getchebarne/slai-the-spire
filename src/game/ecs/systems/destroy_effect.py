from src.game.ecs.components.effects import EffectApplyToComponent
from src.game.ecs.manager import ECSManager
from src.game.ecs.systems.base import BaseSystem
from src.game.ecs.systems.base import ProcessStatus


class DestroyEffect(BaseSystem):
    def process(self, manager: ECSManager) -> ProcessStatus:
        effect_entity_id, (draw_card_effect_component, effect_apply_to_component) = next(
            manager.get_components(EffectApplyToComponent)
        )
        manager.destroy_entity(effect_entity_id)

        return ProcessStatus.COMPLETE
