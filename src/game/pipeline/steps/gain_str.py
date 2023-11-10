from game.effects.base import BaseEffect
from game.pipeline.steps.base import BaseStep


class GainStrength(BaseStep):
    @property
    def priority(self) -> int:
        return 1

    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.target.buffs.strength.increase_stack(effect.plus_str)

    def _condition(self, effect: BaseEffect) -> bool:
        return effect.plus_str is not None
