import numpy as np
import torch
from slai import IntentKind
from slai import Monster
from slai import MonsterName

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.encoding.modifier import ENCODING_DIM_MODIFIERS
from src.rl.encoding.modifier import encode_modifiers_into
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm


# Order = fill order = Core's global-offset order
SLICE_MONSTERS = [Slice(SliceKind.MONSTERS, MAX_MONSTERS)]

# int() keying: declaration-order discriminant, stable across engine bumps
_MAP_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
_MAP_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
_INTENT_BLOCK_KINDS = {IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff}
_INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
_INTENT_DEBUFF_KINDS = {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}

# Normalization caps, bump when adding heavy hitters
_DAMAGE_MAX = 96  # Bronze Automaton's Hyper Beam 50 x `FACTOR_VULN` 1.5 = 75
_INSTANCES_MAX = 5  # The Guardian's Whirlwind
_HEALTH_MAX = HEALTH_MAX_ENTITY  # Champ 440 at A9+ (constants.py)
_BLOCK_MAX = BLOCK_MAX_ENTITY  # Spheric Guardian lives at 40 (constants.py)
_STOLEN_GOLD_MAX = 40  # Mugger/Looter thievery accumulation; sqrt-scaled
_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)

_DIM_HEALTH_BLOCK = get_encoding_dim_health_block(_HEALTH_MAX, _BLOCK_MAX)

ENCODING_DIM_MONSTER = (
    ENCODING_DIM_MODIFIERS  # Modifiers OHE
    + _DIM_HEALTH_BLOCK  # Health and block OHE and scalars
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
    + 1  # Stolen gold (Mugger/Looter; returned on kill)
    + len(_MAP_INTENT_KIND)  # Intent kind OHE
)

# Family-block positions (consumed by EntityProjector's shared first layer). The
# balance assert ties the slices to _encode_monster_into's write order.
SLICE_MODIFIERS_MONSTER = slice(0, ENCODING_DIM_MODIFIERS)
SLICE_HEALTH_BLOCK_MONSTER = slice(
    ENCODING_DIM_MODIFIERS, ENCODING_DIM_MODIFIERS + _DIM_HEALTH_BLOCK
)
_NUM_TAIL_SCALARS = 10  # the scalar block in _encode_monster_into
assert SLICE_HEALTH_BLOCK_MONSTER.stop == ENCODING_DIM_MONSTER - (
    len(_MAP_MONSTER_NAME) + _DAMAGE_DIM + _NUM_TAIL_SCALARS + len(_MAP_INTENT_KIND)
), "monster row layout drifted: family slices no longer match ENCODING_DIM_MONSTER"


def _intent_total_damage(monster: Monster) -> int:
    """Monster's total attack damage this turn: per-hit damage * instances (0 if not attacking)."""
    return (monster.intent.damage or 0) * (monster.intent.instances or 1)


def _encode_monster_into(
    monster: Monster, char_health: int, char_block: int, outgoing_damage: float, out: np.ndarray
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
    total_damage = _intent_total_damage(monster)
    out[pos] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
    out[pos + 1] = min(monster.intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
    out[pos + 2] = float(monster.intent.kind in _INTENT_BLOCK_KINDS)
    out[pos + 3] = float(monster.intent.kind in _INTENT_BUFF_KINDS)
    out[pos + 4] = float(monster.intent.kind in _INTENT_DEBUFF_KINDS)
    out[pos + 5] = float(total_damage <= char_block)
    out[pos + 6] = float(outgoing_damage >= char_health + char_block)
    out[pos + 7] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[pos + 8] = monster.health / max(monster.health_max, 1)
    out[pos + 9] = get_sqrt_norm(monster.gold_stolen, _STOLEN_GOLD_MAX)
    pos += 10

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
    np_out = np.zeros((batch_size, MAX_MONSTERS, ENCODING_DIM_MONSTER), dtype=np.float32)
    np_pad = np.zeros((batch_size, MAX_MONSTERS), dtype=bool)
    outgoing_damages = []

    for b, monsters in enumerate(batch_monster):
        # Total incoming this turn = sum over attackers (turn-lethal flag; also for character).
        outgoing_damage = sum(_intent_total_damage(monster) for monster in monsters)

        for i, monster in enumerate(monsters):
            _encode_monster_into(
                monster, batch_health[b], batch_block[b], outgoing_damage, np_out[b, i]
            )
            np_pad[b, i] = True

        outgoing_damages.append(outgoing_damage)

    return (
        torch.from_numpy(np_out).to(device),
        torch.from_numpy(np_pad).to(device),
        outgoing_damages,
    )
