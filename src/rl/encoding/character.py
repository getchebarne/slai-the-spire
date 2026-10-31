import math
from enum import StrEnum
from enum import auto

import numpy as np
from slai import CardCostKindMinusDiscardsThisTurn
from slai import GameEnv
from slai import GameState
from slai import get_card_templates

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STOLEN_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
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


# Health / Block is not here: it is the combatants' health/block columns (health_block.py)
class CharacterChannel(StrEnum):
    GOLD = auto()
    GOLD_SCALAR = auto()
    HEALTH_FRACTION = auto()
    MAX_HEALTH_MAGNITUDE = auto()
    DECK_SIZE = auto()
    DECK_UPGRADED_FRACTION = auto()
    REWARD_ROLL_OFFSET = auto()
    POTION_DROP_CHANCE = auto()  # Combat Potion odds, ignoring White Beast Statue's 100%
    SHOP_PURGE_COST_RUN = auto()  # Card removal price before Relic discounts

    # Combat only: 0 outside it
    INCOMING_DAMAGE = auto()
    BLOCK_COVERS_INCOMING = auto()
    INCOMING_IS_LETHAL = auto()
    INCOMING_HIDDEN = auto()  # Runic Dome: the four incoming-damage channels read 0
    NET_UNBLOCKED_DAMAGE = auto()
    THIS_TURN_DISCARDS = auto()
    THIS_TURN_ATTACKS = auto()  # Finisher hits once per Attack
    THIS_TURN_CARDS_PLAYED = auto()  # Normality caps the turn at 3
    PANACHE_COUNTDOWN = auto()  # Plays left until Panache's hit
    ORANGE_PELLETS_PLAYED_ATTACK = auto()  # Kinds played this turn since Orange Pellets fired
    ORANGE_PELLETS_PLAYED_SKILL = auto()
    ORANGE_PELLETS_PLAYED_POWER = auto()
    TURN_END_STARTED = auto()  # From the turn end's first pass to the Character's next turn start
    TURN_END_PENDING = auto()  # From Time Warp's 12th play until the turn end's first pass starts
    TURN_ENDED = auto()  # From the first Monster turn start to the Character's next turn start
    BOSS_FIGHT = auto()  # A Boss-kind Monster was in the opening roster
    THIS_COMBAT_THIEF_ESCAPED = auto()  # A Smoke Bomb then still keeps the Reward
    THIS_COMBAT_MONSTER_DIED = auto()  # Escapes then no longer cost the gold and the Potion roll
    GOLD_STOLEN = auto()  # The killed thieves' gold, returned as its own Reward item


CHANNEL_LAYOUT_CHARACTER, _alloc = make_channel_allocator()
OFF_CHARACTER_GOLD = _alloc(CharacterChannel.GOLD, _GOLD_DIM)
OFF_CHARACTER_GOLD_SCALAR = _alloc(CharacterChannel.GOLD_SCALAR)
OFF_CHARACTER_HEALTH_FRACTION = _alloc(CharacterChannel.HEALTH_FRACTION)
OFF_CHARACTER_MAX_HEALTH_MAGNITUDE = _alloc(CharacterChannel.MAX_HEALTH_MAGNITUDE)
OFF_CHARACTER_DECK_SIZE = _alloc(CharacterChannel.DECK_SIZE)
OFF_CHARACTER_DECK_UPGRADED_FRACTION = _alloc(CharacterChannel.DECK_UPGRADED_FRACTION)
OFF_CHARACTER_REWARD_ROLL_OFFSET = _alloc(CharacterChannel.REWARD_ROLL_OFFSET)
OFF_CHARACTER_POTION_DROP_CHANCE = _alloc(CharacterChannel.POTION_DROP_CHANCE)
OFF_CHARACTER_SHOP_PURGE_COST_RUN = _alloc(CharacterChannel.SHOP_PURGE_COST_RUN)
OFF_CHARACTER_INCOMING_DAMAGE = _alloc(CharacterChannel.INCOMING_DAMAGE)
OFF_CHARACTER_BLOCK_COVERS_INCOMING = _alloc(CharacterChannel.BLOCK_COVERS_INCOMING)
OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc(CharacterChannel.INCOMING_IS_LETHAL)
OFF_CHARACTER_INCOMING_HIDDEN = _alloc(CharacterChannel.INCOMING_HIDDEN)
OFF_CHARACTER_NET_UNBLOCKED_DAMAGE = _alloc(CharacterChannel.NET_UNBLOCKED_DAMAGE)
OFF_CHARACTER_THIS_TURN_DISCARDS = _alloc(CharacterChannel.THIS_TURN_DISCARDS)
OFF_CHARACTER_THIS_TURN_ATTACKS = _alloc(CharacterChannel.THIS_TURN_ATTACKS)
OFF_CHARACTER_THIS_TURN_CARDS_PLAYED = _alloc(CharacterChannel.THIS_TURN_CARDS_PLAYED)
OFF_CHARACTER_PANACHE_COUNTDOWN = _alloc(CharacterChannel.PANACHE_COUNTDOWN)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_ATTACK)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_SKILL)
OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER = _alloc(CharacterChannel.ORANGE_PELLETS_PLAYED_POWER)
OFF_CHARACTER_TURN_END_STARTED = _alloc(CharacterChannel.TURN_END_STARTED)
OFF_CHARACTER_TURN_END_PENDING = _alloc(CharacterChannel.TURN_END_PENDING)
OFF_CHARACTER_TURN_ENDED = _alloc(CharacterChannel.TURN_ENDED)
OFF_CHARACTER_BOSS_FIGHT = _alloc(CharacterChannel.BOSS_FIGHT)
OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED = _alloc(CharacterChannel.THIS_COMBAT_THIEF_ESCAPED)
OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED = _alloc(CharacterChannel.THIS_COMBAT_MONSTER_DIED)
OFF_CHARACTER_GOLD_STOLEN = _alloc(CharacterChannel.GOLD_STOLEN)

ENCODING_DIM_CHARACTER = sum(w for _, w in CHANNEL_LAYOUT_CHARACTER)

# Most nonzero columns a Character row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_CHARACTER = len(CHANNEL_LAYOUT_CHARACTER)


def encode_character_into(
    game_state: GameState, incoming_damage: int | None, out: np.ndarray
) -> None:
    character = game_state.character
    deck = game_state.card_pile_deck
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
        min(max(GameEnv.POTION_DROP_CHANCE_BASE + game_state.potion_drop_mod, 0), 100) / 100
    )
    out[OFF_CHARACTER_SHOP_PURGE_COST_RUN] = (
        min(game_state.shop_purge_cost_run, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    )
    combat = game_state.combat
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
        out[OFF_CHARACTER_THIS_TURN_ATTACKS] = get_sqrt_norm(
            combat.this_turn_attacks, MAX_SIZE_CARD_PILE_DECK
        )
        out[OFF_CHARACTER_THIS_TURN_CARDS_PLAYED] = get_sqrt_norm(
            combat.this_turn_cards_played, MAX_SIZE_CARD_PILE_DECK
        )
        out[OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / _PANACHE_PLAYS
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK] = float(
            combat.orange_pellets_played_attack
        )
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL] = float(combat.orange_pellets_played_skill)
        out[OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER] = float(combat.orange_pellets_played_power)
        out[OFF_CHARACTER_TURN_END_STARTED] = float(combat.turn_end_started)
        out[OFF_CHARACTER_TURN_END_PENDING] = float(combat.turn_end_pending)
        out[OFF_CHARACTER_TURN_ENDED] = float(combat.turn_ended)
        out[OFF_CHARACTER_BOSS_FIGHT] = float(combat.boss_fight)
        out[OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED] = float(combat.this_combat_thief_escaped)
        out[OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED] = float(combat.this_combat_monster_died)
        out[OFF_CHARACTER_GOLD_STOLEN] = get_sqrt_norm(combat.gold_stolen, GOLD_STOLEN_MAX)
