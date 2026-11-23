import math

import numpy as np
from slai import CARD_REWARD_ROLL_OFFSET_BASE
from slai import MAX_NORMALITY_PLAYS
from slai import MAX_VELVET_CHOKER_PLAYS
from slai import MIN_CARD_REWARD_ROLL_OFFSET
from slai import POTION_DROP_CHANCE_BASE
from slai import Card
from slai import CardCostKindMinusDiscardsThisTurn
from slai import CardName
from slai import Combat
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
from src.rl.encoding.shared import encode_shared_health_block_into
from src.rl.tokens import get_character
from src.rl.tokens import get_entities
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
_REWARD_ROLL_OFFSET_MIN = MIN_CARD_REWARD_ROLL_OFFSET
_REWARD_ROLL_OFFSET_BASE = CARD_REWARD_ROLL_OFFSET_BASE

# This turn's discards only cut Eviscerate's cost and unlock Sneaky Strike's refund
_THIS_TURN_DISCARDS_MAX = max(
    card_template.cost_base
    for card_template in get_card_templates()
    if isinstance(card_template.cost_kind, CardCostKindMinusDiscardsThisTurn)
)

# Panache hits on every 5th play of a turn: the engine's PANACHE_PLAYS
# (src/consts.rs), which GameEnv doesn't expose
_PANACHE_PLAYS = 5

_THIS_TURN_PLAYS_LEFT_DIM = MAX_VELVET_CHOKER_PLAYS + 1


# Health / Block is not here: it is the combatants' health/block columns (health_block.py)


_CHANNEL_LAYOUT_CHARACTER, _alloc = make_channel_allocator()
_OFF_CHARACTER_GOLD = _alloc("gold", _GOLD_DIM)
_OFF_CHARACTER_GOLD_SCALAR = _alloc("gold_scalar")
_OFF_CHARACTER_HEALTH_FRACTION = _alloc("health_fraction")
_OFF_CHARACTER_MAX_HEALTH_MAGNITUDE = _alloc("max_health_magnitude")
_OFF_CHARACTER_DECK_SIZE = _alloc("deck_size")
_OFF_CHARACTER_DECK_UPGRADED_FRACTION = _alloc("deck_upgraded_fraction")
_OFF_CHARACTER_REWARD_ROLL_OFFSET = _alloc("reward_roll_offset")
_OFF_CHARACTER_POTION_DROP_CHANCE = _alloc("potion_drop_chance")
_OFF_CHARACTER_SHOP_PURGE_COST_RUN = _alloc("shop_purge_cost_run")
_OFF_CHARACTER_POTION_SLOTS_LEFT = _alloc("potion_slots_left")
# Combat only: 0 outside it
_OFF_CHARACTER_INCOMING_DAMAGE = _alloc("incoming_damage")
_OFF_CHARACTER_INCOMING_DAMAGE_OHE = _alloc("incoming_damage_ohe", _INCOMING_DIM)
_OFF_CHARACTER_INCOMING_HEALTH_LOST = _alloc("incoming_health_lost")
_OFF_CHARACTER_INCOMING_HEALTH_LOST_OHE = _alloc("incoming_health_lost_ohe", _INCOMING_DIM)
_OFF_CHARACTER_INCOMING_IS_LETHAL = _alloc("incoming_is_lethal")
_OFF_CHARACTER_INCOMING_HIDDEN = _alloc("incoming_hidden")
_OFF_CHARACTER_THIS_TURN_DISCARDS = _alloc("this_turn_discards")
_OFF_CHARACTER_THIS_TURN_ATTACKS = _alloc("this_turn_attacks")
_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED = _alloc("this_turn_cards_played")
_OFF_CHARACTER_THIS_TURN_PLAYS_LEFT = _alloc("this_turn_plays_left", _THIS_TURN_PLAYS_LEFT_DIM)
_OFF_CHARACTER_PANACHE_COUNTDOWN = _alloc("panache_countdown")
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK = _alloc("orange_pellets_played_attack")
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL = _alloc("orange_pellets_played_skill")
_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER = _alloc("orange_pellets_played_power")
_OFF_CHARACTER_TURN_END_STARTED = _alloc("turn_end_started")
_OFF_CHARACTER_TURN_END_PENDING = _alloc("turn_end_pending")
_OFF_CHARACTER_TURN_ENDED = _alloc("turn_ended")
_OFF_CHARACTER_BOSS_FIGHT = _alloc("boss_fight")
_OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED = _alloc("this_combat_thief_escaped")
_OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED = _alloc("this_combat_monster_died")
_OFF_CHARACTER_GOLD_STOLEN = _alloc("gold_stolen")

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
        The Combat, for the hand's ids and the turn's plays.

    Returns
    -------
    int | None
        The plays left under the tightest cap, Normality's while it's in hand, else Velvet
        Choker's; None with neither.
    """
    hand = get_entities(game_state, combat.id_card_pile_hand, Card)
    if any(card.name == CardName.Normality for card in hand):
        plays_max = MAX_NORMALITY_PLAYS
    elif relic_held(game_state, RelicName.VelvetChoker):
        plays_max = MAX_VELVET_CHOKER_PLAYS
    else:
        return None

    return max(plays_max - combat.this_turn_cards_played, 0)


def encode_character_into(
    game_state: GameState, incoming: Incoming | None, np_own: np.ndarray, np_shared: np.ndarray
) -> None:
    """Write the Character's token: its own columns, then its health/block.

    Parameters
    ----------
    game_state : GameState
        The state to encode.
    incoming : Incoming | None
        What ending the turn now would cost (incoming.py); None outside Combat or when Runic Dome
        hides the intents.
    np_own : np.ndarray
        (ENCODING_DIM_CHARACTER,) float32. The Character's own range of its token row, written
        in place.
    np_shared : np.ndarray
        (ENCODING_DIM_SHARED,) float32. The token's shared block, written in place.
    """
    character = get_character(game_state)
    deck = get_entities(game_state, game_state.id_card_pile_deck, Card)
    gold_bucket = get_piecewise_bucket(
        character.gold, _GOLD_MIN, _GOLD_MAX, _GOLD_LINEAR_SQRT_THRESHOLD
    )
    np_own[_OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    np_own[_OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, _GOLD_MAX) / _GOLD_MAX
    np_own[_OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    np_own[_OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(
        character.health_max, _HEALTH_MAX_CAP
    )
    np_own[_OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_CARD_PILE_DECK)
    np_own[_OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )
    np_own[_OFF_CHARACTER_REWARD_ROLL_OFFSET] = (
        character.reward_roll_offset - _REWARD_ROLL_OFFSET_MIN
    ) / (_REWARD_ROLL_OFFSET_BASE - _REWARD_ROLL_OFFSET_MIN)
    np_own[_OFF_CHARACTER_POTION_DROP_CHANCE] = (
        min(max(POTION_DROP_CHANCE_BASE + game_state.potion_drop_mod, 0), 100) / 100
    )
    np_own[_OFF_CHARACTER_SHOP_PURGE_COST_RUN] = (
        min(game_state.shop_purge_cost_run, SHOP_PRICE_CAP) / SHOP_PRICE_CAP
    )
    np_own[_OFF_CHARACTER_POTION_SLOTS_LEFT] = (
        game_state.potion_slots_max - len(game_state.id_potions)
    ) / MAX_POTION_SLOTS
    combat = game_state.combat
    if combat is not None:
        if incoming is None:
            np_own[_OFF_CHARACTER_INCOMING_HIDDEN] = 1.0
        else:
            np_own[_OFF_CHARACTER_INCOMING_DAMAGE] = get_sqrt_norm(
                incoming.damage, _INCOMING_DAMAGE_MAX
            )
            incoming_damage_bucket = get_piecewise_bucket(
                incoming.damage, 0, _INCOMING_DAMAGE_MAX, _INCOMING_LINEAR_SQRT_THRESHOLD
            )
            np_own[_OFF_CHARACTER_INCOMING_DAMAGE_OHE + incoming_damage_bucket] = 1.0
            np_own[_OFF_CHARACTER_INCOMING_HEALTH_LOST] = get_sqrt_norm(
                incoming.health_lost, _INCOMING_DAMAGE_MAX
            )
            incoming_health_lost_bucket = get_piecewise_bucket(
                incoming.health_lost, 0, _INCOMING_DAMAGE_MAX, _INCOMING_LINEAR_SQRT_THRESHOLD
            )
            np_own[_OFF_CHARACTER_INCOMING_HEALTH_LOST_OHE + incoming_health_lost_bucket] = 1.0
            np_own[_OFF_CHARACTER_INCOMING_IS_LETHAL] = float(incoming.is_lethal)
        np_own[_OFF_CHARACTER_THIS_TURN_DISCARDS] = (
            min(combat.this_turn_discards, _THIS_TURN_DISCARDS_MAX) / _THIS_TURN_DISCARDS_MAX
        )
        np_own[_OFF_CHARACTER_THIS_TURN_ATTACKS] = get_sqrt_norm(
            combat.this_turn_attacks, MAX_SIZE_CARD_PILE_DECK
        )
        np_own[_OFF_CHARACTER_THIS_TURN_CARDS_PLAYED] = get_sqrt_norm(
            combat.this_turn_cards_played, MAX_SIZE_CARD_PILE_DECK
        )
        this_turn_plays_left = _get_this_turn_plays_left(game_state, combat)
        if this_turn_plays_left is not None:
            np_own[_OFF_CHARACTER_THIS_TURN_PLAYS_LEFT + this_turn_plays_left] = 1.0
        np_own[_OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / _PANACHE_PLAYS
        np_own[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK] = float(
            combat.orange_pellets_played_attack
        )
        np_own[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL] = float(
            combat.orange_pellets_played_skill
        )
        np_own[_OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER] = float(
            combat.orange_pellets_played_power
        )
        np_own[_OFF_CHARACTER_TURN_END_STARTED] = float(combat.turn_end_started)
        np_own[_OFF_CHARACTER_TURN_END_PENDING] = float(combat.turn_end_pending)
        np_own[_OFF_CHARACTER_TURN_ENDED] = float(combat.turn_ended)
        np_own[_OFF_CHARACTER_BOSS_FIGHT] = float(combat.boss_fight)
        np_own[_OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED] = float(combat.this_combat_thief_escaped)
        np_own[_OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED] = float(combat.this_combat_monster_died)
        np_own[_OFF_CHARACTER_GOLD_STOLEN] = get_sqrt_norm(combat.gold_stolen, GOLD_STOLEN_MAX)

    # Health & Block
    encode_shared_health_block_into(character.health, character.block, np_shared)
