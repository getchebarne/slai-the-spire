import numpy as np
import torch
from slai import IntentKind
from slai import Monster
from slai import MonsterName
from slai import members

from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.encoding.modifier import encode_modifiers_into
from src.rl.encoding.modifier import get_encoding_dim_modifiers
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm

_MAP_MONSTER_NAME = {name: i for i, name in enumerate(members(MonsterName))}
_MAP_INTENT_KIND = {intent_kind: i for i, intent_kind in enumerate(members(IntentKind))}
_INTENT_BLOCK_KINDS = {IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff}
_INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
_INTENT_DEBUFF_KINDS = {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}

# Normalization caps, bump when adding heavy hitters
_DAMAGE_MAX = 64  # post-scaling: Guardian/Slime Boss base ~36-38 x FACTOR_VULN 1.5 ~= 54-57
_INSTANCES_MAX = 5  # The Guardian's Whirlwind
_HEALTH_MAX = 250  # The Guardian's max health
_BLOCK_MAX = 35

_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)

ENCODING_DIM_MONSTER = (
    get_encoding_dim_modifiers()  # Modifiers OHE
    + get_encoding_dim_health_block(_HEALTH_MAX, _BLOCK_MAX)  # Health and block OHE and scalars
    + len(_MAP_MONSTER_NAME)  # Name OHE
    + _DAMAGE_DIM  # Intent damage OHE
    + 1  # Intent damage scalar
    + 1  # Intent instances
    + 1  # Intent has block
    + 1  # Intent has buff
    + 1  # Intent has debuff
    + 1  # Hit fully blocked
    + 1  # Hit is lethal
    + 1  # Max-HP magnitude
    + 1  # Health fraction of max
    + len(_MAP_INTENT_KIND)  # Intent kind OHE
)


def _encode_monster_into(
    monster: Monster, char_health: int, char_block: int, out: np.ndarray
) -> None:
    # Initialize current position pointer
    pos = 0

    # Modifiers OHE
    pos = encode_modifiers_into(monster.modifiers, pos, out)

    # Health and block OHE and scalars
    pos = encode_health_block_into(
        monster.health, monster.block, _HEALTH_MAX, _BLOCK_MAX, pos, out
    )

    # Per-monster-name one-hot
    out[pos + _MAP_MONSTER_NAME[monster.name]] = 1.0
    pos += len(_MAP_MONSTER_NAME)

    # Intent damage OHE
    damage = monster.intent.damage or 0
    damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
    out[pos + damage_bucket] = 1.0
    pos += _DAMAGE_DIM

    # Scalars
    total_damage = damage * (monster.intent.instances or 1)
    out[pos] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
    out[pos + 1] = min(monster.intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
    out[pos + 2] = float(monster.intent.kind in _INTENT_BLOCK_KINDS)
    out[pos + 3] = float(monster.intent.kind in _INTENT_BUFF_KINDS)
    out[pos + 4] = float(monster.intent.kind in _INTENT_DEBUFF_KINDS)
    out[pos + 5] = float(total_damage <= char_block)
    out[pos + 6] = float(total_damage >= char_health + char_block)
    out[pos + 7] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[pos + 8] = monster.health / max(monster.health_max, 1)
    pos += 9

    # Intent kind OHE (the category flags above miss Escape/Sleep/Stunned/Unknown)
    out[pos + _MAP_INTENT_KIND[monster.intent.kind]] = 1.0


def encode_batch_monsters(
    batch_monster: list[list[Monster]],
    batch_health: list[int],
    batch_block: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    batch_size = len(batch_monster)

    # Pre-allocate NumPy arrays
    x_out = np.zeros((batch_size, MAX_MONSTERS, ENCODING_DIM_MONSTER), dtype=np.float32)
    x_pad = np.zeros((batch_size, MAX_MONSTERS), dtype=bool)
    outgoing_damages = []

    for b, monsters in enumerate(batch_monster):
        outgoing_damage = 0.0

        for i, monster in enumerate(monsters):
            _encode_monster_into(monster, batch_health[b], batch_block[b], x_out[b, i])
            x_pad[b, i] = True
            outgoing_damage += (monster.intent.damage or 0.0) * (monster.intent.instances or 1.0)

        outgoing_damages.append(outgoing_damage)

    return (
        torch.from_numpy(x_out).to(device),
        torch.from_numpy(x_pad).to(device),
        outgoing_damages,
    )
