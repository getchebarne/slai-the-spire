import math
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
from slai import get_monster_templates

from src.rl.constants import BLOCK_MAX_ENTITY
from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STEAL_MAX
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import MAX_MONSTERS
from src.rl.constants import MONSTER_MOVESET_MAX
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

# Normalization caps, catalog-anchored (constants.py folds over the Monster templates)
_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # widest hit, Vulnerable-scaled
_INSTANCES_MAX = INTENT_INSTANCES_MAX
_HEALTH_MAX = HEALTH_MAX_ENTITY  # Champ 440 at A9+ (constants.py)
_BLOCK_MAX = BLOCK_MAX_ENTITY  # Spheric Guardian lives at 40 (constants.py)

# Thievery accumulation: Mug, Mug, Lunge — every attack before the Escape turn steals
_STOLEN_GOLD_MAX = 3 * GOLD_STEAL_MAX
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
    MOVE_CURRENT = auto()  # index into the spawn-rolled moveset
    MOVE_PREV_1 = auto()  # one move back (move_history's last entry is the current move)
    MOVE_PREV_2 = auto()


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
OFF_MONSTER_MOVE_CURRENT = _alloc(MonsterChannel.MOVE_CURRENT, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_1 = _alloc(MonsterChannel.MOVE_PREV_1, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_2 = _alloc(MonsterChannel.MOVE_PREV_2, MONSTER_MOVESET_MAX)

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

    # Move identity: the current move + the two before it (history's last entry
    # IS the current move, so the previous ones sit at [-2] and [-3])
    if monster.move_current is not None:
        out[OFF_MONSTER_MOVE_CURRENT + monster.move_current] = 1.0
    history = monster.move_history
    if len(history) >= 2:
        out[OFF_MONSTER_MOVE_PREV_1 + history[-2]] = 1.0
    if len(history) >= 3:
        out[OFF_MONSTER_MOVE_PREV_2 + history[-3]] = 1.0


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


# Whole-moveset Effects per (ascension, MonsterName) — static capability context,
# resolved once from the templates (the spawn-rolled variants are flattened)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, list[Effect]]] = {}


def _moveset_effects(ascension: int) -> dict[MonsterName, list[Effect]]:
    by_name = _MOVESET_EFFECTS.get(ascension)
    if by_name is None:
        by_name = {
            template.name: [effect for move in template.moves for effect in move.effects]
            for template in get_monster_templates(ascension)
        }
        _MOVESET_EFFECTS[ascension] = by_name
    return by_name


def iter_moveset_effect_owners(game_state: GameState) -> Iterator[tuple[int, list[Effect]]]:
    """(slot, whole-moveset Effects) per Monster — what it could do, not the live intent."""
    combat = game_state.combat
    if combat is None:
        return
    by_name = _moveset_effects(game_state.ascension)
    for i, monster in enumerate(combat.monsters[:MAX_MONSTERS]):
        effects = by_name[monster.name]
        if effects:
            yield i, effects
