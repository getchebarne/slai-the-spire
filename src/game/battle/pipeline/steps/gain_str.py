from game.battle.pipeline.steps.base import BaseStep
from game.effects.base import BaseEffect


class GainStrength(BaseStep):
    @property
    def priority(self) -> int:
        return 1

    def _apply_effect(self, effect: BaseEffect) -> None:
        effect.target.buffs.strength += effect.plus_str

    def _condition(self, effect: BaseEffect) -> bool:
        return bool(effect.plus_str)
