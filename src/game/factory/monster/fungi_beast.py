import random

from src.game.core.effect import Effect
from src.game.core.effect import EffectTargetType
from src.game.core.effect import EffectType
from src.game.entity.actor import ModifierData
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster


HEALTH_MAX_MIN = 24
HEALTH_MAX_MAX = 28
BITE_DAMAGE = 6
GROW_STRENGTH = 5
SPORE_CLOUD_STACKS_CURRENT = 2
SPORE_CLOUD_STACKS_MIN = 1
SPORE_CLOUD_STACKS_MAX = 999
SPORE_CLOUD_STACKS_DURATION = False


def _get_effects_bite() -> list[Effect]:
    return [Effect(EffectType.DAMAGE_DEAL, BITE_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_grow() -> list[Effect]:
    return [
        Effect(EffectType.MODIFIER_STRENGTH_GAIN, GROW_STRENGTH, EffectTargetType.SOURCE),
    ]


def create_monster_fungi_beast() -> EntityMonster:
    # TODO: add ascension parameter
    health_max = random.randint(HEALTH_MAX_MIN, HEALTH_MAX_MAX)
    health_current = health_max

    return EntityMonster(
        "Fungi Beast",
        health_current,
        health_max,
        move_map={
            "Bite": _get_effects_bite(),
            "Grow": _get_effects_grow(),
        },
        modifier_map={
            ModifierType.SPORE_CLOUD: ModifierData(
                SPORE_CLOUD_STACKS_CURRENT,
                SPORE_CLOUD_STACKS_MIN,
                SPORE_CLOUD_STACKS_MAX,
                SPORE_CLOUD_STACKS_DURATION,
            )
        },
    )
