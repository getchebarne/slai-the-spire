import random

from src.game.combat.ai import ais
from src.game.combat.effect_queue import process_queue
from src.game.combat.state import Effect
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.utils import add_effects_to_bot


def combat_start(state: GameState) -> None:
    # Shuffle deck into draw pile
    state.card_in_draw_pile_ids = list(state.card_in_deck_ids)
    random.shuffle(state.card_in_draw_pile_ids)

    # Get first move from monsters. TODO: revisit
    for monster in state.get_monsters():
        ais[monster.name](monster)

    # TODO: relic start of combat effects


def _turn_start_actor(state: GameState, actor_id: int) -> None:
    state.actor_turn_id = actor_id

    # Zero block
    add_effects_to_bot(state, Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.TURN))


def turn_start_character(state: GameState) -> None:
    _turn_start_actor(state, state.character_id)

    # Draw 5 cards and refill energy
    add_effects_to_bot(state, Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY))

    # Run effects
    process_queue(state)


def turn_start_monster(state: GameState, monster_id: int) -> None:
    _turn_start_actor(state, monster_id)

    # Run effects
    process_queue(state)


def turn_monster(state: GameState, monster_id: int) -> None:
    monster = state.get_entity(monster_id)
    add_effects_to_bot(state, *monster.move.effects)

    # Run effects
    process_queue(state)


def turn_end_monster(state: GameState, monster_id: int) -> None:
    # Update monster's move
    monster = state.get_entity(monster_id)
    ais[monster.name](monster)


def turn_end_character(state: GameState) -> None:
    # Discard hand
    add_effects_to_bot(
        state,
        Effect(
            EffectType.DISCARD,
            target_type=EffectTargetType.CARD_IN_HAND,
            selection_type=EffectSelectionType.ALL,
        ),
    )

    # Run effects
    process_queue(state)
