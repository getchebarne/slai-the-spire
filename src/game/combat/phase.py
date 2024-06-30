import random

from src.game.combat.context import Actor
from src.game.combat.context import Effect
from src.game.combat.context import EffectSelectionType
from src.game.combat.context import EffectTargetType
from src.game.combat.context import EffectType
from src.game.combat.context import GameContext
from src.game.combat.effect_queue import process_queue
from src.game.combat.utils import add_effects_to_bot


def combat_start(context: GameContext) -> None:
    # Shuffle deck into draw pile
    context.draw_pile = list(context.deck)
    random.shuffle(context.draw_pile)

    # TODO: relic start of combat effects


def _turn_start_actor(context: GameContext, actor: Actor) -> None:
    context.turn = actor

    # Zero block
    add_effects_to_bot(context, Effect(EffectType.ZERO_BLOCK, target_type=EffectTargetType.TURN))


def turn_start_character(context: GameContext) -> None:
    _turn_start_actor(context, context.character)

    # Draw 5 cards and refill energy
    add_effects_to_bot(context, Effect(EffectType.DRAW_CARD, 5), Effect(EffectType.REFILL_ENERGY))

    # Run effects
    process_queue(context)


def turn_end_character(context: GameContext) -> None:
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
