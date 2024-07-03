import random

from src.game.combat.ai import ais
from src.game.combat.effect_queue import process_queue
from src.game.combat.state import Character
from src.game.combat.state import Effect
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import Monster
from src.game.combat.utils import add_effects_to_bot


def combat_start(state: GameState) -> None:
    # Shuffle deck into draw pile
    state.card_in_draw_pile_ids = list(state.card_in_deck_ids)
    random.shuffle(state.card_in_draw_pile_ids)

    # Get first move from monsters. TODO: revisit
    for monster in state.get_monsters():
        ais[monster.name](monster)

    # Set start of turn to character & call it's turn start
    state.actor_turn_id = state.character_id
    _turn_start_effects(state, state.character_id)

    # TODO: relic start of combat effects


def turn_switch(state: GameState) -> None:
    actor_turn_id = state.actor_turn_id
    actor = state.get_entity(actor_turn_id)

    # Process turn end effects
    _turn_end_effects(state, actor_turn_id)

    # Pass "turn" state to next actor
    if isinstance(actor, Character):
        state.actor_turn_id = state.monster_ids[0]

    elif isinstance(actor, Monster):
        # Get monster's position
        monster_pos = state.monster_ids.index(actor_turn_id)

        # If it's the last monster, give "turn_start" to the character
        if monster_pos == len(state.monster_ids) - 1:
            state.actor_turn_id = state.character_id

        # Else, give it to the next monster
        else:
            state.actor_turn_id = state.monster_ids[monster_pos + 1]

    else:
        raise ValueError(f"Unsupported actor instance: {actor.__class__.__name__}")

    # Process turn start effects
    _turn_start_effects(state, state.actor_turn_id)


def _turn_start_effects(state: GameState, actor_id: int) -> None:
    # Common effects
    effects = [(Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.TURN), None)]

    # Character and monster-specific effects
    actor = state.get_entity(actor_id)
    if isinstance(actor, Character):
        # Draw 5 cards and refill energy
        effects += [
            (Effect(EffectType.DRAW_CARD, 5), None),
            (Effect(EffectType.REFILL_ENERGY), None),
        ]

    elif isinstance(actor, Monster):
        # TODO: empty for now
        pass

    # Process effects
    add_effects_to_bot(state, *effects)
    process_queue(state)


def _turn_end_effects(state: GameState, actor_id: int) -> None:
    actor = state.get_entity(actor_id)

    # Common
    for modifier_type, modifier in actor.modifiers.items():
        if modifier.stacks_duration:
            modifier.stacks -= 1

    actor.modifiers = {
        modifier_type: modifier
        for modifier_type, modifier in actor.modifiers.items()
        if modifier.stacks > modifier.stacks_min
    }

    if isinstance(actor, Character):
        # Character-specific effects
        effects = [
            (
                Effect(
                    EffectType.DISCARD,
                    target_type=EffectTargetType.CARD_IN_HAND,
                    selection_type=EffectSelectionType.ALL,
                ),
                None,
            )
        ]

    elif isinstance(actor, Monster):
        # TODO: no effects for now
        effects = []

    # Process effects
    add_effects_to_bot(state, *effects)
    process_queue(state)


def turn_monster(state: GameState) -> None:
    actor_id = state.actor_turn_id
    actor = state.get_entity(actor_id)

    if not isinstance(actor, Monster):
        return

    # Queue monster's move's effects
    add_effects_to_bot(state, *[(effect, actor_id) for effect in actor.move.effects])

    # Update monster's move
    ais[actor.name](actor)

    # Run effects
    process_queue(state)

    # Switch turn
    turn_switch(state)
