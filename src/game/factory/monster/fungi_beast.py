import random

from src.game.combat.effect import Effect
from src.game.combat.effect import EffectTargetType
from src.game.combat.effect import EffectType
from src.game.entity.actor import ModifierType
from src.game.entity.monster import EntityMonster
from src.game.factory.modifier.spore_cloud import create_modifier_spore_cloud


HEALTH_MAX_MIN = 24
HEALTH_MAX_MAX = 28
BITE_DAMAGE = 6
GROW_STRENGTH = 5
STACKS_CURRENT_SPORE_CLOUD = 2


def _get_effects_bite() -> list[Effect]:
    return [Effect(EffectType.DEAL_DAMAGE, BITE_DAMAGE, EffectTargetType.CHARACTER)]


def _get_effects_grow() -> list[Effect]:
    return [
        Effect(EffectType.GAIN_STRENGTH, GROW_STRENGTH, EffectTargetType.SOURCE),
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
            ModifierType.SPORE_CLOUD: create_modifier_spore_cloud(STACKS_CURRENT_SPORE_CLOUD)
        },
    )
