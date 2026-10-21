import math
from enum import StrEnum
from enum import auto

import numpy as np
import torch
from slai import Card
from slai import CardCostKindMinusDiscardsThisTurn
from slai import Character
from slai import Combat
from slai import GameEnv
from slai import GameState
from slai import get_card_templates

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import FACTOR_VULN
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.encoding.monster import outgoing_damage
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.encoding.sparse import token_sparse
from src.rl.types import TokenKind
from src.rl.types import X_Tokens
from src.rl.types import make_channel_allocator


# Widest catalog turn (damage x instances), Vulnerable-scaled; Strength ramps clamp
_INCOMING_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)
_HEALTH_MAX_CAP = CHARACTER_HEALTH_CAP
_GOLD_MIN = 0
_GOLD_MAX = 999
_GOLD_LINEAR_SQRT_THRESHOLD = _GOLD_MIN  # Pure sqrt
_GOLD_DIM = get_piecewise_dim(_GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD)

# Card-reward pity: starts at the base, falls by one per Common down to the min, resets on a Rare
_REWARD_ROLL_OFFSET_MIN = GameEnv.CARD_REWARD_ROLL_OFFSET_MIN
_REWARD_ROLL_OFFSET_BASE = GameEnv.CARD_REWARD_ROLL_OFFSET_BASE

# This turn's discards only cut Eviscerate's cost and unlock Sneaky Strike's refund
_THIS_TURN_DISCARDS_MAX = max(
    card_template.cost_base
    for card_template in get_card_templates()
    if isinstance(card_template.cost_kind, CardCostKindMinusDiscardsThisTurn)
)

# Panache hits on every 5th play of a turn: the engine's PANACHE_PLAYS
# (src/consts.rs), which GameEnv doesn't expose
_PANACHE_PLAYS = 5


# Health / Block is not here: it is the combatants' shared family (health_block.py)
class CharacterChannel(StrEnum):
    GOLD = auto()
    GOLD_SCALAR = auto()
    HEALTH_FRACTION = auto()
    MAX_HEALTH_MAGNITUDE = auto()
    DECK_SIZE = auto()
    DECK_UPGRADED_FRACTION = auto()
    REWARD_ROLL_OFFSET = auto()
    POTION_DROP_CHANCE = auto()  # Combat potion odds, ignoring White Beast Statue's 100%

    # Combat only: 0 outside it
    INCOMING_DAMAGE = auto()
    BLOCK_COVERS_INCOMING = auto()
    INCOMING_IS_LETHAL = auto()
    INCOMING_HIDDEN = auto()  # Runic Dome: the four incoming-damage channels read 0
    NET_UNBLOCKED_DAMAGE = auto()
    THIS_TURN_DISCARDS = auto()
    PANACHE_COUNTDOWN = auto()  # Plays left until Panache's hit
    ORANGE_PELLETS_PLAYED_ATTACK = auto()  # Kinds played this turn since Orange Pellets fired
    ORANGE_PELLETS_PLAYED_SKILL = auto()
    ORANGE_PELLETS_PLAYED_POWER = auto()


CHANNEL_LAYOUT_CHARACTER, _alloc = make_channel_allocator()
OFF_CHARACTER_GOLD = _alloc(CharacterChannel.GOLD, _GOLD_DIM)
OFF_CHARACTER_GOLD_SCALAR = _alloc(CharacterChannel.GOLD_SCALAR)
OFF_CHARACTER_HEALTH_FRACTION = _alloc(CharacterChannel.HEALTH_FRACTION)
OFF_CHARACTER_MAX_HEALTH_MAGNITUDE = _alloc(CharacterChannel.MAX_HEALTH_MAGNITUDE)
OFF_CHARACTER_DECK_SIZE = _alloc(CharacterChannel.DECK_SIZE)
OFF_CHARACTER_DECK_UPGRADED_FRACTION = _alloc(CharacterChannel.DECK_UPGRADED_FRACTION)
OFF_CHARACTER_REWARD_ROLL_OFFSET = _alloc(CharacterChannel.REWARD_ROLL_OFFSET)
OFF_CHARACTER_POTION_DROP_CHANCE = _alloc(CharacterChannel.POTION_DROP_CHANCE)
OFF_CHARACTER_INCOMING_DAMAGE = _alloc(CharacterChannel.INCOMING_DAMAGE)
OFF_CHARACTER_BLOCK_COVERS_INCOMING = _alloc(CharacterChannel.BLOCK_COVERS_INCOMING)
OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc(CharacterChannel.INCOMING_IS_LETHAL)
OFF_CHARACTER_INCOMING_HIDDEN = _alloc(CharacterChannel.INCOMING_HIDDEN)
OFF_CHARACTER_NET_UNBLOCKED_DAMAGE = _alloc(CharacterChannel.NET_UNBLOCKED_DAMAGE)
OFF_CHARACTER_THIS_TURN_DISCARDS = _alloc(CharacterChannel.THIS_TURN_DISCARDS)
OFF_CHARACTER_PANACHE_COUNTDOWN = _alloc(CharacterChannel.PANACHE_COUNTDOWN)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_ATTACK)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_SKILL)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_POWER)

ENCODING_DIM_CHARACTER = sum(w for _, w in CHANNEL_LAYOUT_CHARACTER)

# Most nonzero columns a character row holds: one per channel, each a one-hot or a scalar
ENCODING_SPARSE_CHARACTER = len(CHANNEL_LAYOUT_CHARACTER)


def _encode_character_into(
    character: Character,
    incoming_damage: int | None,
    deck: list[Card],
    combat: Combat | None,
    potion_drop_mod: int,
    out: np.ndarray,
) -> None:
    gold_bucket = get_piecewise_bucket(
        character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD
    )
    out[OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    out[OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, _GOLD_MAX) / _GOLD_MAX
    out[OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    out[OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(character.health_max, _HEALTH_MAX_CAP)
    out[OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_CARD_PILE_DECK)
    out[OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )
    out[OFF_CHARACTER_REWARD_ROLL_OFFSET] = (
        character.reward_roll_offset - _REWARD_ROLL_OFFSET_MIN
    ) / (_REWARD_ROLL_OFFSET_BASE - _REWARD_ROLL_OFFSET_MIN)
    out[OFF_CHARACTER_POTION_DROP_CHANCE] = (
        min(max(GameEnv.POTION_DROP_CHANCE_BASE + potion_drop_mod, 0), 100) / 100
    )
    if combat is not None:
        if incoming_damage is None:
            out[OFF_CHARACTER_INCOMING_HIDDEN] = 1.0
        else:
            out[OFF_CHARACTER_INCOMING_DAMAGE] = get_sqrt_norm(
                incoming_damage, _INCOMING_DAMAGE_MAX
            )
            out[OFF_CHARACTER_BLOCK_COVERS_INCOMING] = float(character.block >= incoming_damage)
            out[OFF_CHARACTER_INCOMING_IS_LETHAL] = float(
                incoming_damage >= character.health + character.block
            )
            out[OFF_CHARACTER_NET_UNBLOCKED_DAMAGE] = get_sqrt_norm(
                max(incoming_damage - character.block, 0), _INCOMING_DAMAGE_MAX
            )
        out[OFF_CHARACTER_THIS_TURN_DISCARDS] = (
            min(combat.this_turn_discards, _THIS_TURN_DISCARDS_MAX) / _THIS_TURN_DISCARDS_MAX
        )
        out[OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / _PANACHE_PLAYS
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK] = float(
            combat.orange_pellets_played_attack
        )
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL] = float(combat.orange_pellets_played_skill)
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER] = float(combat.orange_pellets_played_power)


def encode_batch_character(
    game_states: list[GameState], column_start: int, device: torch.device
) -> X_Tokens:
    """The character tokens: one row per state, (B, dim) held sparse, its columns from
    `column_start` in the shared space, + the (B,) token kind — the singleton is always present —
    and each state's (B,) count, 1."""
    batch_size = len(game_states)

    # Pre-allocate NumPy array
    np_out = np.zeros((batch_size, ENCODING_DIM_CHARACTER), dtype=np.float32)

    for b, game_state in enumerate(game_states):
        combat = game_state.combat
        monsters = combat.monsters if combat is not None else []

        _encode_character_into(
            game_state.character,
            outgoing_damage(monsters),
            game_state.card_pile_deck,
            combat,
            game_state.potion_drop_mod,
            np_out[b],
        )

    np_token_kind = np.full(batch_size, TokenKind.CHARACTER, dtype=np.int16)
    np_count = np.ones(batch_size, dtype=np.int64)
    return token_sparse(
        np_out, np_token_kind, np_count, ENCODING_SPARSE_CHARACTER, column_start, device
    )
