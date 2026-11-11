import math

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
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm


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


_CHANNEL_LAYOUT_CHARACTER, _alloc = make_channel_allocator()
_OFF_CHARACTER_GOLD = _alloc("gold", _GOLD_DIM)
_OFF_CHARACTER_GOLD_SCALAR = _alloc("gold_scalar")
_OFF_CHARACTER_HEALTH_FRACTION = _alloc("health_fraction")
_OFF_CHARACTER_MAX_HEALTH_MAGNITUDE = _alloc("max_health_magnitude")
_OFF_CHARACTER_DECK_SIZE = _alloc("deck_size")
_OFF_CHARACTER_DECK_UPGRADED_FRACTION = _alloc("deck_upgraded_fraction")
_OFF_CHARACTER_REWARD_ROLL_OFFSET = _alloc("reward_roll_offset")
_OFF_CHARACTER_POTION_DROP_CHANCE = _alloc(
    "potion_drop_chance"
)  # Combat Potion odds, ignoring White Beast Statue's 100%
_OFF_CHARACTER_SHOP_PURGE_COST_RUN = _alloc(
    "shop_purge_cost_run"
)  # Card removal price before Relic discounts
# Combat only: 0 outside it
_OFF_CHARACTER_INCOMING_DAMAGE = _alloc("incoming_damage")
_OFF_CHARACTER_BLOCK_COVERS_INCOMING = _alloc("block_covers_incoming")
_OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc("incoming_is_lethal")
_OFF_CHARACTER_INCOMING_HIDDEN = _alloc(
    "incoming_hidden"
)  # Runic Dome: the four incoming-damage channels read 0
_OFF_CHARACTER_NET_UNBLOCKED_DAMAGE = _alloc("net_unblocked_damage")
_OFF_CHARACTER_THIS_TURN_DISCARDS = _alloc("this_turn_discards")
_OFF_CHARACTER_THIS_TURN_ATTACKS = _alloc("this_turn_attacks")  # Finisher hits once per Attack
_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED = _alloc(
    "this_turn_cards_played"
)  # Normality caps the turn at 3
_OFF_CHARACTER_PANACHE_COUNTDOWN = _alloc("panache_countdown")  # Plays left until Panache's hit
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK = _alloc(
    "orange_pellets_played_attack"
)  # Kinds played this turn since Orange Pellets fired
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL = _alloc("orange_pellets_played_skill")
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER = _alloc("orange_pellets_played_power")
_OFF_CHARACTER_TURN_END_STARTED = _alloc(
    "turn_end_started"
)  # From the turn end's first pass to the Character's next turn start
_OFF_CHARACTER_TURN_END_PENDING = _alloc(
    "turn_end_pending"
)  # From Time Warp's 12th play until the turn end's first pass starts
_OFF_CHARACTER_TURN_ENDED = _alloc(
    "turn_ended"
)  # From the first Monster turn start to the Character's next turn start
_OFF_CHARACTER_BOSS_FIGHT = _alloc("boss_fight")  # A Boss-kind Monster was in the opening roster
_OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED = _alloc(
    "this_combat_thief_escaped"
)  # A Smoke Bomb then still keeps the Reward
_OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED = _alloc(
    "this_combat_monster_died"
)  # Escapes then no longer cost the gold and the Potion roll
_OFF_CHARACTER_GOLD_STOLEN = _alloc(
    "gold_stolen"
)  # The killed thieves' gold, returned as its own Reward item

ENCODING_DIM_CHARACTER = sum(w for _, w in _CHANNEL_LAYOUT_CHARACTER)

# Most nonzero columns a Character row holds: one per channel, each a one-hot or a scalar
ENCODING_DIM_SPARSE_CHARACTER = len(_CHANNEL_LAYOUT_CHARACTER)


def encode_character_into(
    game_state: GameState, incoming_damage: int | None, out: np.ndarray
) -> None:
    character = game_state.character
    deck = game_state.card_pile_deck
    gold_bucket = get_piecewise_bucket(
        character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD
    )
    out[_OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    out[_OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, _GOLD_MAX) / _GOLD_MAX
    out[_OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    out[_OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(character.health_max, _HEALTH_MAX_CAP)
    out[_OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_CARD_PILE_DECK)
    out[_OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )
    out[_OFF_CHARACTER_REWARD_ROLL_OFFSET] = (
        character.reward_roll_offset - _REWARD_ROLL_OFFSET_MIN
    ) / (_REWARD_ROLL_OFFSET_BASE - _REWARD_ROLL_OFFSET_MIN)
    out[_OFF_CHARACTER_POTION_DROP_CHANCE] = (
        min(max(GameEnv.POTION_DROP_CHANCE_BASE + game_state.potion_drop_mod, 0), 100) / 100
    )
    out[_OFF_CHARACTER_SHOP_PURGE_COST_RUN] = (
        min(game_state.shop_purge_cost_run, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    )
    combat = game_state.combat
    if combat is not None:
        if incoming_damage is None:
            out[_OFF_CHARACTER_INCOMING_HIDDEN] = 1.0
        else:
            out[_OFF_CHARACTER_INCOMING_DAMAGE] = get_sqrt_norm(
                incoming_damage, _INCOMING_DAMAGE_MAX
            )
            out[_OFF_CHARACTER_BLOCK_COVERS_INCOMING] = float(character.block >= incoming_damage)
            out[_OFF_CHARACTER_INCOMING_IS_LETHAL] = float(
                incoming_damage >= character.health + character.block
            )
            out[_OFF_CHARACTER_NET_UNBLOCKED_DAMAGE] = get_sqrt_norm(
                max(incoming_damage - character.block, 0), _INCOMING_DAMAGE_MAX
            )
        out[_OFF_CHARACTER_THIS_TURN_DISCARDS] = (
            min(combat.this_turn_discards, _THIS_TURN_DISCARDS_MAX) / _THIS_TURN_DISCARDS_MAX
        )
        out[_OFF_CHARACTER_THIS_TURN_ATTACKS] = get_sqrt_norm(
            combat.this_turn_attacks, MAX_SIZE_CARD_PILE_DECK
        )
        out[_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED] = get_sqrt_norm(
            combat.this_turn_cards_played, MAX_SIZE_CARD_PILE_DECK
        )
        out[_OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / _PANACHE_PLAYS
        out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK] = float(
            combat.orange_pellets_played_attack
        )
        out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL] = float(combat.orange_pellets_played_skill)
        out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER] = float(combat.orange_pellets_played_power)
        out[_OFF_CHARACTER_TURN_END_STARTED] = float(combat.turn_end_started)
        out[_OFF_CHARACTER_TURN_END_PENDING] = float(combat.turn_end_pending)
        out[_OFF_CHARACTER_TURN_ENDED] = float(combat.turn_ended)
        out[_OFF_CHARACTER_BOSS_FIGHT] = float(combat.boss_fight)
        out[_OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED] = float(combat.this_combat_thief_escaped)
        out[_OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED] = float(combat.this_combat_monster_died)
        out[_OFF_CHARACTER_GOLD_STOLEN] = get_sqrt_norm(combat.gold_stolen, GOLD_STOLEN_MAX)
