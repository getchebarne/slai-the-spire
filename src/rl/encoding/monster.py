import math
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

from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STEAL_MAX
from src.rl.constants import HEALTH_MAX_ENTITY
from src.rl.constants import INTENT_DAMAGE_HIT_MAX
from src.rl.constants import INTENT_INSTANCES_MAX
from src.rl.constants import MONSTER_MOVESET_MAX
from src.rl.types import EffectSource
from src.rl.types import SliceKind
from src.rl.types import make_channel_allocator
from src.rl.utils import get_piecewise_bucket
from src.rl.utils import get_piecewise_dim
from src.rl.utils import get_sqrt_norm


_MAP_MONSTER_NAME = {name: int(name) for name in MonsterName.members()}
_MAP_INTENT_KIND = {intent_kind: int(intent_kind) for intent_kind in IntentKind.members()}
_INTENT_BLOCK_KINDS = {IntentKind.Block, IntentKind.AttackBlock, IntentKind.BlockBuff}
_INTENT_BUFF_KINDS = {IntentKind.Buff, IntentKind.AttackBuff, IntentKind.BlockBuff}
_INTENT_DEBUFF_KINDS = {IntentKind.Debuff, IntentKind.AttackDebuff, IntentKind.DebuffPowerful}

# Normalization caps, catalog-anchored (constants.py folds over the Monster templates)
_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_HIT_MAX * FACTOR_VULN)  # widest hit, Vulnerable-scaled
_INSTANCES_MAX = INTENT_INSTANCES_MAX
_HEALTH_MAX = HEALTH_MAX_ENTITY  # Champ 440 at A9+ (constants.py)

# Thievery accumulation: Mug, Mug, Lunge — every attack before the Escape turn steals
_STOLEN_GOLD_MAX = 3 * GOLD_STEAL_MAX
_LINEAR_SQRT_THRESHOLD = 18
_DAMAGE_DIM = get_piecewise_dim(0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)


# Health/block is not here: it is the combatants' shared family (health_block.py)
class MonsterChannel(StrEnum):
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
    INTENT_HIDDEN = auto()  # Runic Dome: every intent channel and hit flag reads 0
    MOVE_CURRENT = auto()  # index into the spawn-rolled moveset
    MOVE_PREV_1 = auto()  # one move back (move_history's last entry is the current move)
    MOVE_PREV_2 = auto()
    HOLDS_STASIS = auto()  # a card it took sits in its Combat.card_pile_stasis slot


CHANNEL_LAYOUT_MONSTER, _alloc = make_channel_allocator()
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
OFF_MONSTER_INTENT_HIDDEN = _alloc(MonsterChannel.INTENT_HIDDEN)
OFF_MONSTER_MOVE_CURRENT = _alloc(MonsterChannel.MOVE_CURRENT, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_1 = _alloc(MonsterChannel.MOVE_PREV_1, MONSTER_MOVESET_MAX)
OFF_MONSTER_MOVE_PREV_2 = _alloc(MonsterChannel.MOVE_PREV_2, MONSTER_MOVESET_MAX)
OFF_MONSTER_HOLDS_STASIS = _alloc(MonsterChannel.HOLDS_STASIS)

ENCODING_DIM_MONSTER = sum(w for _, w in CHANNEL_LAYOUT_MONSTER)


def _intent_total_damage(monster: Monster) -> int:
    intent = monster.intent
    if intent is None:  # Runic Dome hides it
        return 0
    return (intent.damage or 0) * (intent.instances or 1)


def _encode_monster_into(
    monster: Monster, char_health: int, char_block: int, outgoing_damage: float, out: np.ndarray
) -> None:
    out[OFF_MONSTER_NAME + _MAP_MONSTER_NAME[monster.name]] = 1.0
    out[OFF_MONSTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(monster.health_max, _HEALTH_MAX)
    out[OFF_MONSTER_HEALTH_FRACTION] = monster.health / max(monster.health_max, 1)
    out[OFF_MONSTER_GOLD_STOLEN] = get_sqrt_norm(monster.gold_stolen, _STOLEN_GOLD_MAX)

    intent = monster.intent
    if intent is None:
        out[OFF_MONSTER_INTENT_HIDDEN] = 1.0
    else:
        damage = intent.damage or 0
        damage_bucket = get_piecewise_bucket(damage, 0, _DAMAGE_MAX, _LINEAR_SQRT_THRESHOLD)
        out[OFF_MONSTER_INTENT_DAMAGE + damage_bucket] = 1.0

        total_damage = _intent_total_damage(monster)
        out[OFF_MONSTER_INTENT_DAMAGE_SCALAR] = min(damage, _DAMAGE_MAX) / _DAMAGE_MAX
        out[OFF_MONSTER_INTENT_INSTANCES] = (
            min(intent.instances or 0, _INSTANCES_MAX) / _INSTANCES_MAX
        )
        out[OFF_MONSTER_INTENT_HAS_BLOCK] = float(intent.kind in _INTENT_BLOCK_KINDS)
        out[OFF_MONSTER_INTENT_HAS_BUFF] = float(intent.kind in _INTENT_BUFF_KINDS)
        out[OFF_MONSTER_INTENT_HAS_DEBUFF] = float(intent.kind in _INTENT_DEBUFF_KINDS)
        out[OFF_MONSTER_HIT_FULLY_BLOCKED] = float(total_damage <= char_block)
        out[OFF_MONSTER_HIT_IS_LETHAL] = float(outgoing_damage >= char_health + char_block)

        # The intent category flags above miss Escape/Sleep/Stunned/Unknown
        out[OFF_MONSTER_INTENT_KIND + _MAP_INTENT_KIND[intent.kind]] = 1.0

    # Move identity: the current move + the two before it (history's last entry
    # IS the current move, so the previous ones sit at [-2] and [-3])
    if monster.move_current is not None:
        out[OFF_MONSTER_MOVE_CURRENT + monster.move_current] = 1.0
    # Under Runic Dome the history holds only the executed moves: no current
    # entry ends it, so the previous ones sit at [-1] and [-2]
    shift = 0 if intent is None else 1
    history = monster.move_history
    if len(history) >= 1 + shift:
        out[OFF_MONSTER_MOVE_PREV_1 + history[-1 - shift]] = 1.0
    if len(history) >= 2 + shift:
        out[OFF_MONSTER_MOVE_PREV_2 + history[-2 - shift]] = 1.0


def outgoing_damage(monsters: list[Monster]) -> int:
    """Total incoming this turn = sum over attackers (turn-lethal flag; also for character)."""
    return sum(_intent_total_damage(monster) for monster in monsters)


def encode_batch_monsters(
    game_states: list[GameState], device: torch.device
) -> tuple[
    torch.Tensor,
    np.ndarray,
    list[list[tuple[int, EffectSource, Effect]]],
    list[list[tuple[int, EffectSource, Effect]]],
]:
    """(B, W, dim) rows + the (B, W) kind of each token + each state's live-move
    effect rows and whole-moveset rows (what it could do, not the live intent),
    both keyed by monster slot; W is the batch's most monsters."""
    batch_size = len(game_states)
    batch_monsters = []
    batch_card_pile_stasis = []
    for game_state in game_states:
        combat = game_state.combat  # hoisted: each read clones across the FFI
        batch_monsters.append(combat.monsters if combat is not None else [])
        batch_card_pile_stasis.append(combat.card_pile_stasis if combat is not None else [])

    width = max((len(monsters) for monsters in batch_monsters), default=0)
    np_out = np.zeros((batch_size, width, ENCODING_DIM_MONSTER), dtype=np.float32)
    np_idxs = np.full((batch_size, width), -1, dtype=np.int16)
    batch_rows_move: list[list[tuple[int, EffectSource, Effect]]] = []
    batch_rows_moveset: list[list[tuple[int, EffectSource, Effect]]] = []

    for b, (game_state, monsters, card_pile_stasis) in enumerate(
        zip(game_states, batch_monsters, batch_card_pile_stasis)
    ):
        character = game_state.character
        damage = outgoing_damage(monsters)
        by_name = _moveset_effects(game_state.ascension)
        rows_move: list[tuple[int, EffectSource, Effect]] = []
        rows_moveset: list[tuple[int, EffectSource, Effect]] = []
        for i, monster in enumerate(monsters):
            _encode_monster_into(monster, character.health, character.block, damage, np_out[b, i])
            # The stasis pile is slot-aligned with the monsters: None where one holds no card
            np_out[b, i, OFF_MONSTER_HOLDS_STASIS] = float(card_pile_stasis[i] is not None)
            np_idxs[b, i] = SliceKind.MONSTERS
            effects = monster.move_effects  # bound once: each read clones the Vec
            rows_move += [(i, EffectSource.MONSTER_MOVE, effect) for effect in effects]
            rows_moveset += [
                (i, EffectSource.MONSTER_MOVESET, effect) for effect in by_name[monster.name]
            ]
        batch_rows_move.append(rows_move)
        batch_rows_moveset.append(rows_moveset)

    return torch.from_numpy(np_out).to(device), np_idxs, batch_rows_move, batch_rows_moveset


# Whole-moveset Effects per (ascension, MonsterName), collected once (the
# spawn-rolled variants are flattened)
_MOVESET_EFFECTS: dict[int, dict[MonsterName, tuple[Effect, ...]]] = {}


def _moveset_effects(ascension: int) -> dict[MonsterName, tuple[Effect, ...]]:
    by_name = _MOVESET_EFFECTS.get(ascension)
    if by_name is None:
        # One entry per distinct move name: counter/spawn-rolled variants
        # collapse to their last (tables order variants ascending, so last =
        # strongest). Book of Stabbing enumerates a move per stab count — raw
        # flattening costs the triangular sum (210 rows at a0)
        by_name = {
            template.name: tuple(
                effect
                for move in {move.name: move for move in template.moves}.values()
                for effect in move.effects
            )
            for template in get_monster_templates(ascension)
        }
        _MOVESET_EFFECTS[ascension] = by_name
    return by_name
