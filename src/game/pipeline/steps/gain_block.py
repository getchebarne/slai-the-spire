from src.game.core.components import BlockComponent
from src.game.core.effect import Effect
from src.game.core.effect import EffectType
from src.game.core.manager import ECSManager
from src.game.pipeline.steps.base import BaseStep


MAX_BLOCK = 999


# TODO: cap block
class GainBlock(BaseStep):
    def _apply_effect(self, manager: ECSManager, target_entity_id: int, effect: Effect) -> None:
        block = effect.value
        manager.get_component_for_entity(target_entity_id, BlockComponent).current += block

    def _condition(self, effect: Effect) -> bool:
        return effect.type == EffectType.BLOCK
