import numpy as np
from slai import GameEnv
from slai import GameState

from src.rl.constants import GOLD_STOLEN_MAX
from src.rl.constants import MAX_SIZE_CARD_PILE_DECK
from src.rl.constants import SHOP_PRICE_CAP
from src.rl.encoding.scalars import get_sqrt_norm
from src.rl.schema.character import GOLD_LINEAR_SQRT_THRESHOLD
from src.rl.schema.character import GOLD_MAX
from src.rl.schema.character import GOLD_MIN
from src.rl.schema.character import HEALTH_MAX_CAP
from src.rl.schema.character import INCOMING_DAMAGE_MAX
from src.rl.schema.character import OFF_CHARACTER_BLOCK_COVERS_INCOMING
from src.rl.schema.character import OFF_CHARACTER_BOSS_FIGHT
from src.rl.schema.character import OFF_CHARACTER_DECK_SIZE
from src.rl.schema.character import OFF_CHARACTER_DECK_UPGRADED_FRACTION
from src.rl.schema.character import OFF_CHARACTER_GOLD
from src.rl.schema.character import OFF_CHARACTER_GOLD_SCALAR
from src.rl.schema.character import OFF_CHARACTER_GOLD_STOLEN
from src.rl.schema.character import OFF_CHARACTER_HEALTH_FRACTION
from src.rl.schema.character import OFF_CHARACTER_INCOMING_DAMAGE
from src.rl.schema.character import OFF_CHARACTER_INCOMING_HIDDEN
from src.rl.schema.character import OFF_CHARACTER_INCOMING_IS_LETHAL
from src.rl.schema.character import OFF_CHARACTER_MAX_HEALTH_MAGNITUDE
from src.rl.schema.character import OFF_CHARACTER_NET_UNBLOCKED_DAMAGE
from src.rl.schema.character import OFF_CHARACTER_ORANGE_PELLETS_PLAYED_ATTACK
from src.rl.schema.character import OFF_CHARACTER_ORANGE_PELLETS_PLAYED_POWER
from src.rl.schema.character import OFF_CHARACTER_ORANGE_PELLETS_PLAYED_SKILL
from src.rl.schema.character import OFF_CHARACTER_PANACHE_COUNTDOWN
from src.rl.schema.character import OFF_CHARACTER_POTION_DROP_CHANCE
from src.rl.schema.character import OFF_CHARACTER_REWARD_ROLL_OFFSET
from src.rl.schema.character import OFF_CHARACTER_SHOP_PURGE_COST_RUN
from src.rl.schema.character import OFF_CHARACTER_THIS_COMBAT_MONSTER_DIED
from src.rl.schema.character import OFF_CHARACTER_THIS_COMBAT_THIEF_ESCAPED
from src.rl.schema.character import OFF_CHARACTER_THIS_TURN_ATTACKS
from src.rl.schema.character import OFF_CHARACTER_THIS_TURN_CARDS_PLAYED
from src.rl.schema.character import OFF_CHARACTER_THIS_TURN_DISCARDS
from src.rl.schema.character import OFF_CHARACTER_TURN_END_PENDING
from src.rl.schema.character import OFF_CHARACTER_TURN_END_STARTED
from src.rl.schema.character import OFF_CHARACTER_TURN_ENDED
from src.rl.schema.character import PANACHE_PLAYS
from src.rl.schema.character import REWARD_ROLL_OFFSET_BASE
from src.rl.schema.character import REWARD_ROLL_OFFSET_MIN
from src.rl.schema.character import THIS_TURN_DISCARDS_MAX
from src.rl.schema.scalars import get_piecewise_bucket


def encode_character_into(
    game_state: GameState, incoming_damage: int | None, out: np.ndarray
) -> None:
    character = game_state.character
    deck = game_state.card_pile_deck
    gold_bucket = get_piecewise_bucket(
        character.gold, GOLD_MIN, GOLD_MAX, GOLD_LINEAR_SQRT_THRESHOLD
    )
    out[OFF_CHARACTER_GOLD + gold_bucket] = 1.0

    out[OFF_CHARACTER_GOLD_SCALAR] = min(character.gold, GOLD_MAX) / GOLD_MAX
    out[OFF_CHARACTER_HEALTH_FRACTION] = character.health / max(character.health_max, 1)
    out[OFF_CHARACTER_MAX_HEALTH_MAGNITUDE] = get_sqrt_norm(character.health_max, HEALTH_MAX_CAP)
    out[OFF_CHARACTER_DECK_SIZE] = get_sqrt_norm(len(deck), MAX_SIZE_CARD_PILE_DECK)
    out[OFF_CHARACTER_DECK_UPGRADED_FRACTION] = sum(card.upgraded for card in deck) / max(
        len(deck), 1
    )
    out[OFF_CHARACTER_REWARD_ROLL_OFFSET] = (
        character.reward_roll_offset - REWARD_ROLL_OFFSET_MIN
    ) / (REWARD_ROLL_OFFSET_BASE - REWARD_ROLL_OFFSET_MIN)
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
                incoming_damage, INCOMING_DAMAGE_MAX
            )
            out[OFF_CHARACTER_BLOCK_COVERS_INCOMING] = float(character.block >= incoming_damage)
            out[OFF_CHARACTER_INCOMING_IS_LETHAL] = float(
                incoming_damage >= character.health + character.block
            )
            out[OFF_CHARACTER_NET_UNBLOCKED_DAMAGE] = get_sqrt_norm(
                max(incoming_damage - character.block, 0), INCOMING_DAMAGE_MAX
            )
        out[OFF_CHARACTER_THIS_TURN_DISCARDS] = (
            min(combat.this_turn_discards, THIS_TURN_DISCARDS_MAX) / THIS_TURN_DISCARDS_MAX
        )
        out[OFF_CHARACTER_THIS_TURN_ATTACKS] = get_sqrt_norm(
            combat.this_turn_attacks, MAX_SIZE_CARD_PILE_DECK
        )
        out[OFF_CHARACTER_THIS_TURN_CARDS_PLAYED] = get_sqrt_norm(
            combat.this_turn_cards_played, MAX_SIZE_CARD_PILE_DECK
        )
        out[OFF_CHARACTER_PANACHE_COUNTDOWN] = combat.panache_countdown / PANACHE_PLAYS
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
