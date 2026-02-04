import math

import torch

from src.game.const import MAX_MONSTERS
from src.game.factory.lib import FACTORY_LIB_MONSTER
from src.game.factory.monster.the_guardian import _FIERCE_BASH_DAMAGE_ASC_4  # TODO: queries
from src.game.factory.monster.the_guardian import _HEALTH_MAX_ASC_9  # TODO: queries
from src.game.factory.monster.the_guardian import _WHIRLWIND_INSTANCES  # TODO: queries
from src.game.view.monster import ViewIntent
from src.game.view.monster import ViewMonster
from src.rl.encoding.actor import encode_view_actor_modifiers
from src.rl.utils import encode_one_hot_list


_BLOCK_MAX = 20  # TODO: revisit
_BLOCK_MIN = 0
_DAMAGE_MAX = _FIERCE_BASH_DAMAGE_ASC_4
_HEALTH_MAX = _HEALTH_MAX_ASC_9
_HEALTH_MIN = 1
_INSTANCES_MAX = _WHIRLWIND_INSTANCES
_MONSTER_NAMES = list(FACTORY_LIB_MONSTER.keys())

# Pre-computed sqrt bounds for one-hot encoding
_SQRT_HEALTH_MIN = int(math.sqrt(_HEALTH_MIN))
_SQRT_HEALTH_MAX = int(math.sqrt(_HEALTH_MAX))
_SQRT_BLOCK_MIN = int(math.sqrt(_BLOCK_MIN))
_SQRT_BLOCK_MAX = int(math.sqrt(_BLOCK_MAX))
_SQRT_HP_BLOCK_MIN = int(math.sqrt(_HEALTH_MIN + _BLOCK_MIN))
_SQRT_HP_BLOCK_MAX = int(math.sqrt(_HEALTH_MAX + _BLOCK_MAX))

# Sqrt bounds for intent damage
_SQRT_DAMAGE_MIN = 0
_SQRT_DAMAGE_MAX = int(math.sqrt(_DAMAGE_MAX))
_SQRT_DAMAGE_DIM = _SQRT_DAMAGE_MAX - _SQRT_DAMAGE_MIN + 1


def _encode_view_monster(view_monster: ViewMonster) -> list[float]:
    idx_name = _MONSTER_NAMES.index(view_monster.name)
    sqrt_health = int(math.sqrt(view_monster.health_current))
    sqrt_block = int(math.sqrt(view_monster.block_current))
    sqrt_hp_block = int(math.sqrt(view_monster.health_current + view_monster.block_current))
    return (
        # Monster name
        encode_one_hot_list(idx_name, 0, len(_MONSTER_NAMES) - 1)
        # Health / one-hot (sqrt compressed)
        + encode_one_hot_list(sqrt_health, _SQRT_HEALTH_MIN, _SQRT_HEALTH_MAX)
        # Block / one-hot (sqrt compressed)
        + encode_one_hot_list(sqrt_block, _SQRT_BLOCK_MIN, _SQRT_BLOCK_MAX)
        # Health + Block / one-hot (sqrt compressed)
        + encode_one_hot_list(sqrt_hp_block, _SQRT_HP_BLOCK_MIN, _SQRT_HP_BLOCK_MAX)
        # Modifiers
        + encode_view_actor_modifiers(view_monster.modifiers)
        # Intent
        + _encode_view_intent(view_monster.intent)
        # Scalars
        + [
            # Health
            view_monster.health_current / _HEALTH_MAX,
            # Block
            view_monster.block_current / _BLOCK_MAX,
            # Health + Block
            (view_monster.health_current + view_monster.block_current)
            / (_HEALTH_MAX + _BLOCK_MAX),
        ]
    )


def _encode_view_intent(view_intent: ViewIntent) -> list[float]:
    damage = view_intent.damage or 0

    # Sqrt one-hot for damage
    sqrt_damage = int(math.sqrt(damage))
    sqrt_damage = max(min(sqrt_damage, _SQRT_DAMAGE_MAX), _SQRT_DAMAGE_MIN)
    damage_sqrt_one_hot = [0.0] * _SQRT_DAMAGE_DIM
    damage_sqrt_one_hot[sqrt_damage - _SQRT_DAMAGE_MIN] = 1.0

    return damage_sqrt_one_hot + [
        damage / _DAMAGE_MAX,  # Scalar for damage
        (view_intent.instances or 0) / _INSTANCES_MAX,
        float(view_intent.block),
        float(view_intent.buff),
        float(view_intent.debuff_powerful),
    ]


def get_encoding_dim_monster() -> int:
    view_monster_dummy = ViewMonster("Dummy", 0, 0, 0, {}, ViewIntent())
    encoding_monster_dummy = _encode_view_monster(view_monster_dummy)
    return len(encoding_monster_dummy)


# TODO: handle these constants better
_ENCODING_MONSTER_PAD = [0.0] * get_encoding_dim_monster()


def encode_batch_view_monsters(
    batch_view_monster: list[list[ViewMonster]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    # Iterate through each list of monsters in the batch
    x_out = []
    x_mask_pad = []
    outgoing_damages = []
    for view_monsters in batch_view_monster:
        encoding_monsters = []
        outgoing_damage = 0.0
        for view_monster in view_monsters:
            encoding_monsters.append(_encode_view_monster(view_monster))
            outgoing_damage += (view_monster.intent.damage or 0.0) * (
                view_monster.intent.instances or 0.0
            )

        # Padding
        num_pad = MAX_MONSTERS - len(encoding_monsters)
        encoding_monsters += [_ENCODING_MONSTER_PAD] * num_pad

        # Append to final tensor
        x_out.append(encoding_monsters)
        x_mask_pad.append([True] * len(view_monsters) + [False] * num_pad)
        outgoing_damages.append(outgoing_damage)

    return (
        torch.tensor(x_out, dtype=torch.float32, device=device),
        torch.tensor(x_mask_pad, dtype=torch.float32, device=device),
        outgoing_damages,
    )
