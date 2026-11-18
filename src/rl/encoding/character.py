import math

import numpy as np
from slai import CardCostKindMinusDiscardsThisTurn
from slai import CardName
from slai import Combat
from slai import GameEnv
from slai import GameState
from slai import RelicName
from slai import get_card_templates

from src.rl.constants import CHARACTER_HEALTH_CAP
from src.rl.constants import FACTOR_VULN
from src.rl.constants import GOLD_STOLEN_MAX
from src.rl.constants import INTENT_DAMAGE_TOTAL_MAX
from src.rl.constants import MAX_POTION_SLOTS
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.incoming import Incoming
from src.rl.encoding.layout import make_channel_allocator
from src.rl.encoding.scalars import get_piecewise_bucket
from src.rl.encoding.scalars import get_piecewise_dim
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.tokens import relic_held


# Widest catalog turn (damage x instances), Vulnerable-scaled; Strength ramps clamp
_INCOMING_DAMAGE_MAX = math.ceil(INTENT_DAMAGE_TOTAL_MAX * FACTOR_VULN)
_INCOMING_LINEAR_SQRT_THRESHOLD = 25
_INCOMING_DIM = get_piecewise_dim(0, _INCOMING_DAMAGE_MAX, _INCOMING_LINEAR_SQRT_THRESHOLD)
_HEALTH_MAX_CAP = CHARACTER_HEALTH_CAP
_GOLD_MIN = 0
_GOLD_MAX = 999
_GOLD_LINEAR_SQRT_THRESHOLD = _GOLD_MIN  # Pure sqrt
_GOLD_DIM = get_piecewise_dim(_GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD)

# Card-reward pity: starts at the base, falls by one per Common down to the min, resets on a Rare
_REWARD_ROLL_OFFSET_MIN = GameEnv.MIN_CARD_REWARD_ROLL_OFFSET
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
_OFF_CHARACTER_POTION_SLOTS_LEFT = _alloc("potion_slots_left")
# Combat only: 0 outside it
_OFF_CHARACTER_INCOMING_DAMAGE = _alloc(
    "incoming_damage"
)  # What ending the turn now lands on block and HP (incoming.py)
_OFF_CHARACTER_INCOMING_DAMAGE_OHE = _alloc("incoming_damage_ohe", _INCOMING_DIM)
_OFF_CHARACTER_INCOMING_HEALTH_LOST = _alloc(
    "incoming_health_lost"
)  # What ending the turn now costs in HP
_OFF_CHARACTER_INCOMING_HEALTH_LOST_OHE = _alloc("incoming_health_lost_ohe", _INCOMING_DIM)
_OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc("incoming_is_lethal")
_OFF_CHARACTER_INCOMING_HIDDEN = _alloc(
    "incoming_hidden"
)  # Runic Dome: the incoming-damage channels read 0
_OFF_CHARACTER_THIS_TURN_DISCARDS = _alloc("this_turn_discards")
_OFF_CHARACTER_THIS_TURN_ATTACKS = _alloc("this_turn_attacks")  # Finisher hits once per Attack
_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED = _alloc(
    "this_turn_cards_played"
)  # Normality caps the turn at 3
_OFF_CHARACTER_THIS_TURN_PLAYS_LEFT = _alloc(
    "this_turn_plays_left", GameEnv.MAX_VELVET_CHOKER_PLAYS + 1
)  # Under Normality's or Velvet Choker's cap; none without a cap
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


def _get_this_turn_plays_left(game_state: GameState, combat: Combat) -> int | None:
    """Return how many more Cards the turn's play cap allows.

    Parameters
    ----------
    game_state : GameState
        The state, for its Relics.
    combat : Combat
        The Combat, for the hand and the turn's plays.

    Returns
    -------
    int | None
        The plays left under the tightest cap, Normality's while it's in hand, else Velvet
        Choker's; None with neither.
    """
    if any(card.name == CardName.Normality for card in combat.card_pile_hand):
        plays_max = GameEnv.MAX_NORMALITY_PLAYS
    elif relic_held(game_state, RelicName.VelvetChoker):
        plays_max = GameEnv.MAX_VELVET_CHOKER_PLAYS
    else:
        return None
    return max(plays_max - combat.this_turn_cards_played, 0)


def encode_character_into(
    game_state: GameState, incoming: Incoming | None, np_out: np.ndarray
) -> None:
    character = game_state.character
    deck = game_state.card_pile_deck
    gold_bucket = get_piecewise_bucket(
        character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD
    )
    np_out[_OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    np_out[_OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, _GOLD_MAX) / _GOLD_MAX
    np_out[_OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    np_out[_OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(
        character.health_max, _HEALTH_MAX_CAP
    )
    np_out[_OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_CARD_PILE_DECK)
    np_out[_OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )
    np_out[_OFF_CHARACTER_REWARD_ROLL_OFFSET] = (
        character.reward_roll_offset - _REWARD_ROLL_OFFSET_MIN
    ) / (_REWARD_ROLL_OFFSET_BASE - _REWARD_ROLL_OFFSET_MIN)
    np_out[_OFF_CHARACTER_POTION_DROP_CHANCE] = (
        min(max(GameEnv.POTION_DROP_CHANCE_BASE + game_state.potion_drop_mod, 0), 100) / 100
    )
    np_out[_OFF_CHARACTER_SHOP_PURGE_COST_RUN] = (
        min(game_state.shop_purge_cost_run, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    )
    np_out[_OFF_CHARACTER_POTION_SLOTS_LEFT] = (
        game_state.potion_slots_max - len(game_state.potions)
    ) / MAX_POTION_SLOTS
    combat = game_state.combat
    if combat is not None:
        if incoming is None:
            np_out[_OFF_CHARACTER_INCOMING_HIDDEN] = 1.0
        else:
            np_out[_OFF_CHARACTER_INCOMING_DAMAGE] = get_sqrt_norm(
                incoming.damage, _INCOMING_DAMAGE_MAX
            )
            incoming_damage_bucket = get_piecewise_bucket(
                incoming.damage, 0, _INCOMING_DAMAGE_MAX, _INCOMING_LINEAR_SQRT_THRESHOLD
            )
            np_out[_OFF_CHARACTER_INCOMING_DAMAGE_OHE + incoming_damage_bucket] = 1.0
            np_out[_OFF_CHARACTER_INCOMING_HEALTH_LOST] = get_sqrt_norm(
                incoming.health_lost, _INCOMING_DAMAGE_MAX
            )
            incoming_health_lost_bucket = get_piecewise_bucket(
                incoming.health_lost, 0, _INCOMING_DAMAGE_MAX, _INCOMING_LINEAR_SQRT_THRESHOLD
            )
            np_out[_OFF_CHARACTER_INCOMING_HEALTH_LOST_OHE + incoming_health_lost_bucket] = 1.0
            np_out[_OFF_CHARACTER_INCOMING_IS_LETHAL] = float(incoming.is_lethal)
        np_out[_OFF_CHARACTER_THIS_TURN_DISCARDS] = (
            min(combat.this_turn_discards, _THIS_TURN_DISCARDS_MAX) / _THIS_TURN_DISCARDS_MAX
        )
        np_out[_OFF_CHARACTER_THIS_TURN_ATTACKS] = get_sqrt_norm(
            combat.this_turn_attacks, MAX_SIZE_CARD_PILE_DECK
        )
        np_out[_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED] = get_sqrt_norm(
            combat.this_turn_cards_played, MAX_SIZE_CARD_PILE_DECK
        )
        this_turn_plays_left = _get_this_turn_plays_left(game_state, combat)
        if this_turn_plays_left is not None:
            np_out[_OFF_CHARACTER_THIS_TURN_PLAYS_LEFT + this_turn_plays_left] = 1.0
        np_out[_OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / _PANACHE_PLAYS
        np_out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK] = float(
            combat.orange_pellets_played_attack
        )
        np_out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL] = float(
            combat.orange_pellets_played_skill
        )
        np_out[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER] = float(
            combat.orange_pellets_played_power
        )
        np_out[_OFF_CHARACTER_TURN_END_STARTED] = float(combat.turn_end_started)
        np_out[_OFF_CHARACTER_TURN_END_PENDING] = float(combat.turn_end_pending)
        np_out[_OFF_CHARACTER_TURN_ENDED] = float(combat.turn_ended)
        np_out[_OFF_CHARACTER_BOSS_FIGHT] = float(combat.boss_fight)
        np_out[_OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED] = float(combat.this_combat_thief_escaped)
        np_out[_OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED] = float(combat.this_combat_monster_died)
        np_out[_OFF_CHARACTER_GOLD_STOLEN] = get_sqrt_norm(combat.gold_stolen, GOLD_STOLEN_MAX)
