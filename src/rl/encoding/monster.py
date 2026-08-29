from collections.abc import Iterator
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Effect
from slai import GameState
from slai import IntentKind
from slai import Monster
from slai import MonsterName

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import MAX_MONSTERS
from src.rl.encoding.health_block import encode_health_block_into
from src.rl.encoding.health_block import get_encoding_dim_health_block
from src.rl.types import Slice
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm


# Order = fill order = Core's global-offset order
SLICE_MONSTERS = [Slice(SliceKind.MONSTERS, MAX_MONSTERS)]


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


class MonsterChannel(StrEnum):
    HEALTH_BLOCK = auto()
    NAME = auto()
    INTENT_DAMAGE = auto()
    INTENT_DAMAGE_SCALAR = auto()
    INTENT_INSTANCES = auto()
    INTENT_HAS_BLOCK = auto()
    INTENT_HAS_BUFF = auto()
    INTENT_HAS_DEBUFF = auto()
    HIT_FULLY_BLOCKED = auto()
    HIT_IS_LETHAL = auto()
    MAX_HEALTH_MAGNITUDE = auto()
    HEALTH_FRACTION = auto()
    GOLD_STOLEN = auto()
    INTENT_KIND = auto()


CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
OFF_MONSTER_HEALTH_BLOCK = _alloc(MonsterChannel.HEALTH_BLOCK, _DIM_HEALTH_BLOCK)
OFF_MONSTER_NAME = _alloc(MonsterChannel.NAME, len(_MAP_MONSTER_NAME))
OFF_MONSTER_INTENT_DAMAGE = _alloc(MonsterChannel.INTENT_DAMAGE, _DAMAGE_DIM)
OFF_MONSTER_INTENT_DAMAGE_SCALAR = _alloc(MonsterChannel.INTENT_DAMAGE_SCALAR)
OFF_MONSTER_INTENT_INSTANCES = _alloc(MonsterChannel.INTENT_INSTANCES)
OFF_MONSTER_INTENT_HAS_BLOCK = _alloc(MonsterChannel.INTENT_HAS_BLOCK)
OFF_MONSTER_INTENT_HAS_BUFF = _alloc(MonsterChannel.INTENT_HAS_BUFF)
OFF_MONSTER_INTENT_HAS_DEBUFF = _alloc(MonsterChannel.INTENT_HAS_DEBUFF)
OFF_MONSTER_HIT_FULLY_BLOCKED = _alloc(MonsterChannel.HIT_FULLY_BLOCKED)
OFF_MONSTER_HIT_IS_LETHAL = _alloc(MonsterChannel.HIT_IS_LETHAL)
OFF_MONSTER_MAX_HEALTH_MAGNITUDE = _alloc(MonsterChannel.MAX_HEALTH_MAGNITUDE)
OFF_MONSTER_HEALTH_FRACTION = _alloc(MonsterChannel.HEALTH_FRACTION)
OFF_MONSTER_GOLD_STOLEN = _alloc(MonsterChannel.GOLD_STOLEN)
OFF_MONSTER_INTENT_KIND = _alloc(MonsterChannel.INTENT_KIND, len(_MAP_INTENT_KIND))

ENCODING_DIM_MONSTER = sum(w for _, w in CHANNEL_LAYOUT_MONSTER)
SLICE_HEALTH_BLOCK_MONSTER = slice(
    OFF_MONSTER_HEALTH_BLOCK, OFF_MONSTER_HEALTH_BLOCK + _DIM_HEALTH_BLOCK
)


def _intent_total_damage(monster: Monster) -> int:
    return (monster.intent.damage or 0) * (monster.intent.instances or 1)


def _encode_monster_into(
    monster: Monster, char_health: int, char_block: int, outgoing_damage: float, out: np.ndarray
) -> None:
    encode_health_block_into(
        monster.health, monster.block, _HEALTH_MAX, _BLOCK_MAX, OFF_MONSTER_HEALTH_BLOCK, out
    )
    out[OFF_MONSTER_NAME + _MAP_MONSTER_NAME[monster.name]] = 1.0

    damage = monster.intent.damage or 0
    damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
    out[OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

    total_damage = _intent_total_damage(monster)
    out[OFF_MONSTER_INTENT_DAMAGE_SCALAR] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
    out[OFF_MONSTER_INTENT_INSTANCES] = (
        min(monster.intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
    )
    out[OFF_MONSTER_INTENT_HAS_BLOCK] = float(monster.intent.kind in _INTENT_BLOCK_KINDS)
    out[OFF_MONSTER_INTENT_HAS_BUFF] = float(monster.intent.kind in _INTENT_BUFF_KINDS)
    out[OFF_MONSTER_INTENT_HAS_DEBUFF] = float(monster.intent.kind in _INTENT_DEBUFF_KINDS)
    out[OFF_MONSTER_HIT_FULLY_BLOCKED] = float(total_damage <= char_block)
    out[OFF_MONSTER_HIT_IS_LETHAL] = float(outgoing_damage >= char_health + char_block)
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    out[OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, _STOLEN_GOLD_MAX)

    # The intent category flags above miss Escape/Sleep/Stunned/Unknown
    out[OFF_MONSTER_INTENT_KIND + _MAP_INTENT_KIND[monster.intent.kind]] = 1.0


def encode_batch_monsters(
    batch_monster: list[list[Monster]],
    batch_health: list[int],
    batch_block: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    batch_size = len(batch_monster)
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


def iter_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, move Effects) for every Monster whose intent is revealed."""
    combat = game_state.combat
    if combat is None:
        return
    for i, monster in enumerate(combat.monsters[:MAX_MONSTERS]):
        effects = monster.move_effects  # bound once: each read clones the Vec
        if effects:
            yield i, effects
