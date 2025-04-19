from src.game.combat.effect import EFFECT_VALUE_PLACEHOLDER_MODIFIER_DATA_CURRENT_STACKS
from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.actor import ModifierData


STACKS_MIN = 1
STACKS_MAX = 999
STACKS_DURATION = False


def _get_effects_death() -> list[Effect]:
    return [
        Effect(
            EffectType.GAIN_VULNERABLE,
            EFFECT_VALUE_PLACEHOLDER_MODIFIER_DATA_CURRENT_STACKS,
            EffectTargetType.CHARACTER,
        )
    ]


def create_modifier_spore_cloud(stacks_current: int) -> ModifierData:
    return ModifierData(
        stacks_current, STACKS_MIN, STACKS_MAX, STACKS_DURATION, effects_death=_get_effects_death()
    )
