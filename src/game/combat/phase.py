import random

from src.game.combat.ai import ais
from src.game.combat.state import Actor
from src.game.combat.state import Effect
from src.game.combat.state import EffectSelectionType
from src.game.combat.state import EffectTargetType
from src.game.combat.state import EffectType
from src.game.combat.state import GameState
from src.game.combat.state import Monster
from src.game.combat.effect_queue import process_queue
from src.game.combat.utils import add_effects_to_bot


def combat_start(context: GameState) -> None:
    # Shuffle deck into draw pile
    context.draw_pile = list(context.deck)
    random.shuffle(context.draw_pile)

    # Get first move from monsters. TODO: revisit
    for monster in context.monsters:
        ais[monster.name](monster)

    # TODO: relic start of combat effects


def _turn_start_actor(context: GameState, actor: Actor) -> None:
    context.turn = actor

    # Zero block
    add_effects_to_bot(context, Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.TURN))


def turn_start_character(context: GameState) -> None:
    _turn_start_actor(context, context.character)

    # Draw 5 cards and refill energy
    add_effects_to_bot(context, Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY))

    # Run effects
    process_queue(context)


def turn_start_monster(context: GameState, monster: Monster) -> None:
    _turn_start_actor(context, monster)

    # Run effects
    process_queue(context)


def turn_monster(context: GameState, monster: Monster) -> None:
    add_effects_to_bot(context, *monster.move.effects)

    # Run effects
    process_queue(context)


def turn_end_monster(context: GameState, monster: Monster) -> None:
    # Update monster's move
    ais[monster.name](monster)


def turn_end_character(context: GameState) -> None:
    # Discard hand
    add_effects_to_bot(
        context,
        Effect(
            EffectType.DISCARD,
            target_type=EffectTargetType.CARD_IN_HAND,
            selection_type=EffectSelectionType.ALL,
        ),
    )

    # Run effects
    process_queue(context)
