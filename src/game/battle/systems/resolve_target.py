from dataclasses import dataclass
from typing import List, Optional

from game.battle.systems.base import BaseSystem
from game.effects.base import TargetType
from game.effects.base import BaseEffect
from game.entities.actors.base import BaseActor


@dataclass
class TargetedEffect:
    effect: BaseEffect
    source: BaseActor
    target: BaseActor


class ResolveTarget(BaseSystem):
    def __call__(
        self,
        effects: List[BaseEffect],
        source: BaseActor,
        target: Optional[BaseActor] = None,
    ) -> List[TargetedEffect]:
        # TODO: add support for TargetType.RANDOM
        targeted_effects = []
        for effect in effects:
            if effect.target_type == TargetType.SELF:
                targeted_effects.append(TargetedEffect(effect, source, source))

            elif effect.target_type == TargetType.ALL_MONSTERS:
                targeted_effects.extend(
                    [
                        TargetedEffect(effect, source, monster)
                        for monster in self.monsters
                    ]
                )

            elif effect.target_type == TargetType.SINGLE:
                if target is None:
                    raise ValueError(
                        "Argument `target` can't be None if an effect that requires targetting"
                    )
                targeted_effects.append(TargetedEffect(effect, source, target))

        return targeted_effects
